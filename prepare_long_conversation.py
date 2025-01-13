from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
import copy
import json
import hashlib
import random
import jsonlines
from tqdm import tqdm
from functools import partial

from openai import OpenAI


client_info = {
    "gemini": {
        "config": dict(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        ),
        "model": "gemini-1.5-flash-002"
    },
    "openai": {
        "config": dict(
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        "model": "gpt-4o-mini-2024-07-18"
    },
    "llama3.3": {
        "config": dict(
            base_url="http://localhost:9012/v1",
            api_key="token-abc123",
        ),
        "model": "Llama-3.3-70B-Instruct"
    }
}
client_cache: dict[str, OpenAI] = {}

def chat_completions(client: str):
    if client not in client_cache:
        client_cache[client] = OpenAI(**client_info[client]["config"])
    return partial(client_cache[client].chat.completions.create, model=client_info[client]["model"])


# Level của dữ liệu
# - Lĩnh vực
#   - Chủ đề
#     - Hội thoại

class PreparationMixin:
    def __init__(self, filepath: str, keys: list[str], model_name: str="gemini", enable_progress=False, use_jsonl=True, auto_save=True):
        self.filepath = filepath
        self.keys = keys
        self.__values: list[dict] = []
        self.enable_progress = enable_progress
        self.use_jsonl = use_jsonl
        self.auto_save = auto_save
        self.model_name = model_name

        if filepath is not None and os.path.exists(filepath) and os.path.isfile(filepath):
            self.load()

    @property
    def values(self):
        return self.__values

    def add_values(self, *values: list):
        self.__values.extend([self.to_tuple(v) for v in values])

    @classmethod
    def to_tuple(cls, value):
        if isinstance(value, (list, tuple)):
            return tuple([cls.to_tuple(v) for v in value])
        if isinstance(value, dict):
            return {cls.to_tuple(k): cls.to_tuple(v) for k, v in value.items()}
        if isinstance(value, set):
            return {cls.to_tuple(v) for v in value}
        if isinstance(value, (int, float, str)):
            return value
        if value is None:
            return value
        raise NotImplementedError(type(value))

    def sort(self):
        if len(self.__values) == 0:
            return
        
        values = sorted(self.__values, key=lambda x: tuple([x[k] for k in self.keys]))
        self.__values = []
        prev = None
        for val in values:
            if val != prev:
                self.__values.append(val)
            prev = val

    def load(self):
        if self.use_jsonl:
            with jsonlines.open(self.filepath) as f:
                values = list(f)
        else:
            with open(self.filepath, encoding="utf8") as f:
                values = json.load(f)
        self.add_values(*values)
        self.sort()

    def save(self):
        self.sort()
        if self.use_jsonl:
            with jsonlines.open(self.filepath, "w") as f:
                f.write_all(self.__values)
        else:
            with open(self.filepath, "w", encoding="utf8") as f:
                json.dump(self.__values, f, indent=4, ensure_ascii=False)

    def get_prompt(self, **kwargs):
        raise NotImplementedError()
    
    def parse_output(self, output: str, **kwargs) -> list[dict] | None:
        """
        Return `None` if parse failed. Otherwise, return list of parsed values
        """
        raise NotImplementedError()
    
    def run(self, steps: int, **kwargs):
        iterator = range(steps)
        if self.enable_progress:
            iterator = tqdm(iterator)

        results = []

        for _ in iterator:
            prompt = self.get_prompt(**kwargs)
            response = chat_completions(self.model_name)(
                n=1,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=random.uniform(0.85, 1.15),
            )
            output = response.choices[0].message.content
            values = self.parse_output(output, **kwargs)
            if values is not None:
                self.add_values(*values)
                results.extend(values)

        if self.auto_save:
            self.save()
        return results
    
    def prepare_input_requests(self, steps: int, **kwargs):
        requests = []
        for _ in range(steps):
            prompt = self.get_prompt(**kwargs)
            requests.append({
                "request": {
                    # "system_instruction": {
                    #     "parts": { "text": None}
                    # },
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": random.uniform(0.6, 1.2),
                        "max_output_tokens": 2048
                    }
                }
            })
        return requests
    

class AspectPreparation(PreparationMixin):
    """
    Each element contains following attribute:
    - aspect: str
    """

    def __init__(self, filepath, model_name = "gemini", enable_progress=False, auto_save=True):
        super().__init__(filepath, ["aspect"], model_name, enable_progress, True, auto_save)

    def get_prompt(self):
        prompt = "Hãy liệt kê 5 lĩnh vực phù hợp để thảo luận chuyên sâu bằng tiếng Việt."
        if len(self.__values) > 0:
            prompt += f"\nDanh sách hiện tại bao gồm: {','.join(self.__values)}. Chú ý không được liệt kê trùng nhau và trùng với các lĩnh vực đã có sẵn."
        prompt += (
            "\nTrả về theo dạng sau:\n"
            f"1. <tên lĩnh vực>\n"
            f"2. <tên lĩnh vực>\n"
            f"3. <tên lĩnh vực>\n"
            f"4. <tên lĩnh vực>\n"
            f"5. <tên lĩnh vực>\n"
        )
        return prompt

    def parse_output(self, output):
        lines = output.split("\n")
        values = []
        for line in lines:
            words = line.split()
            if len(words) > 1 and words[0] in ["1.", "2.", "3.", "4.", "5."]:
                values.append({"aspect": " ".join(words[1:])})
        return values


class TopicPreparation(PreparationMixin):
    """
    Each element contains following attribute:
    - aspect: str
    - topic: str
    - description
    - meaning
    """

    prompt_list= [
        "Hãy liệt kê 5 chủ đề hấp dẫn và mang tính thời sự trong lĩnh vực {aspect}. Mỗi chủ đề nên có một mô tả ngắn gọn về nội dung và ý nghĩa của nó.",
        "Hãy liệt kê 5 chủ đề gần gũi và mang tính đời thường trong lĩnh vực {aspect}. Mỗi chủ đề nên kèm theo một mô tả ngắn giải thích lý do tại sao chủ đề đó quan trọng trong cuộc sống hàng ngày.",
    ]

    def __init__(self, filepath, model_name = "gemini", enable_progress=False, auto_save=True):
        super().__init__(filepath, ["aspect", "topic"], model_name, enable_progress, True, auto_save)

    def get_prompt(self, aspect: str, prompt_version: int):
        prompt = self.prompt_list[prompt_version].format(aspect=aspect)
        if len(self.__values) > 0:
            prompt += f"\nDanh sách hiện tại bao gồm: {','.join([v['topic'] for v in self.__values])}. Chú ý không được liệt kê trùng nhau và trùng với các chủ đề đã có sẵn."
        prompt += (
            "\nTrả về theo dạng sau:\n"
            
            f"1. **<tên chủ đề>**\n"
            "- Mô tả: <mô tả chủ đề>\n"
            "- Ý nghĩa: <ý nghĩa của chủ đề>\n"
            
            f"2. **<tên chủ đề>**\n"
            "- Mô tả: <mô tả chủ đề>\n"
            "- Ý nghĩa: <ý nghĩa của chủ đề>\n"
            
            f"3. **<tên chủ đề>**\n"
            "- Mô tả: <mô tả chủ đề>\n"
            "- Ý nghĩa: <ý nghĩa của chủ đề>\n"
            
            f"4. **<tên chủ đề>**\n"
            "- Mô tả: <mô tả chủ đề>\n"
            "- Ý nghĩa: <ý nghĩa của chủ đề>\n"

            f"5. **<tên chủ đề>**\n"
            "- Mô tả: <mô tả chủ đề>\n"
            "- Ý nghĩa: <ý nghĩa của chủ đề>\n"
        )
        return prompt
    
    def parse_output(self, output: str, aspect: str, prompt_version: int):
        lines = output.split("\n")
        values = []
        item = None

        def is_valid(item: dict):
            if "topic" in item and "description" in item and "meaning" in item:
                return True
            return False

        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue

            words = line.split()
            if words[0] in ["1.", "2.", "3.", "4.", "5."]:
                if item is not None and is_valid(item):
                    item["batch_id"] = len(values)
                    item["output"] = output
                    values.append(item)
                item = {"aspect": aspect, "prompt_ver": prompt_version, "topic": " ".join(words[1:])}
            elif line.startswith("- Mô tả:"):
                if item is not None and "topic" in item:
                    item["description"] = line[len("- Mô tả:") :].strip()
                else:
                    item = None
            elif line.startswith("- Ý nghĩa:"):
                if item is not None and "topic" in item and "description" in item:
                    item["meaning"] = line[len("- Ý nghĩa:") :].strip()
                else:
                    item = None
        if item is not None and is_valid(item):
            item["batch_id"] = len(values)
            item["output"] = output
            values.append(item)
        return values


class CharacteristicPreparation:
    """
    Kịch bản: 15 câu mỗi người
    - Ví dụ 1:
        - câu 1-5: tích cực
        - câu 6-8: tiêu cực
        - câu 9-15: tích cực
    - Ví dụ 2:
        - câu 1-12: tích cực
        - câu 13-15: tiêu cực
    - Ví dụ 3:
        - câu 1-3: tiêu cực
        - câu 4-15: tích cực
    """

    positive_characteristics = [
        "Bạn tham gia cuộc thảo luận với tâm thế xây dựng, luôn sẵn sàng đóng góp ý kiến một cách tích cực, tập trung vào việc giải quyết vấn đề thay vì chỉ trích.",
        "Bạn tham gia cuộc thảo luận với thái độ cởi mở, chấp nhận các ý kiến mới và sẵn sàng điều chỉnh quan điểm của mình nếu cần.",
        "Bạn tham gia cuộc thảo luận với thái độ tôn trọng, lắng nghe người khác một cách chân thành, thể hiện sự tôn trọng với các quan điểm khác nhau, ngay cả khi không đồng ý.",
        "Bạn tham gia cuộc thảo luận với thái độ tích cực lắng nghe, chú ý đến từng chi tiết trong ý kiến của người khác và phản hồi một cách phù hợp, cho thấy sự quan tâm và thấu hiểu.",
        "Bạn tham gia cuộc thảo luận với tâm thế luôn khuyến khích sự sáng tạo và sự sâu sắc bằng cách đặt những câu hỏi mở, giúp mở rộng góc nhìn trong thảo luận.",
        "Bạn tham gia cuộc thảo luận với tâm thế luôn tìm kiếm cơ hội để làm việc nhóm, hòa nhập và thúc đẩy sự đồng thuận giữa các thành viên.",
        "Bạn tham gia cuộc thảo luận với thái độ rõ ràng và minh bạch, trình bày quan điểm một cách mạch lạc, dễ hiểu để người khác dễ dàng tiếp nhận.",
        "Bạn tham gia cuộc thảo luận với thái độ cầu tiến, sẵn sàng ghi nhận và cải thiện bản thân qua các phản hồi từ người khác.",
    ]

    negative_characteristics = [
        "Bạn tham gia cuộc thảo luận với thái độ thờ ơ, tỏ ra không quan tâm đến nội dung cuộc thảo luận, không đóng góp ý kiến hoặc tham gia một cách miễn cưỡng..",
        "Bạn tham gia cuộc thảo luận với thái độ tiêu cực, phản bác ý kiến của người khác một cách thiếu tôn trọng hoặc mang tính công kích.",
        "Bạn tham gia cuộc thảo luận với thái độ bảo thủ, phản ứng thái quá khi có ý kiến trái chiều hoặc khi bị người khác góp ý, khiến không khí thảo luận trở nên căng thẳng.",
        "Bạn tham gia cuộc thảo luận với thái độ phán xét, hay đánh giá ý kiến của người khác dựa trên thành kiến hoặc cảm xúc cá nhân, thay vì dựa vào nội dung thực tế.",
        "Bạn tham gia cuộc thảo luận với thái độ thiếu chuẩn bị, tham gia thảo luận mà không tìm hiểu trước về chủ đề, dẫn đến ý kiến mơ hồ hoặc thiếu trọng tâm."
    ]

    # Câu chuyển đổi từ positive sang negative (thay thế hoàn toàn tính cách cũ)
    pos_to_neg = "Sau khi thảo luận được một lúc, bạn cảm thấy chưa hài lòng và quyết định chuyển sang thái độ tiêu cực. Lúc này, " # + negative_characteristic

    # Câu chuyển đổi từ negative sang poositive (thay thế hoàn toàn tính cách cũ)
    neg_to_pos = "Sau khi thảo luận được một lúc, bạn nhận ra lỗi sai của mình và quyết định chuyển sang thái độ tích cực hơn. Lúc này, " # + positive_characteristic

    # Nếu gặp đối tác có thái độ tiêu cực thì phải thêm câu này vào prompt
    deal_with_neg = "Chú ý nếu người bạn thảo luận có thái độ tiêu cực, hãy bình tĩnh giải thích cho họ hiểu được vấn đề."

    POSITIVE = 0
    NEGATIVE = 1
    
    def __init__(self, max_turns: int=12):
        self.max_turns = max_turns

    def status(self, characteristic: str):
        if characteristic in self.positive_characteristics:
            return self.POSITIVE
        elif characteristic in self.negative_characteristics:
            return self.NEGATIVE
        raise ValueError(characteristic)

    def get_positive(self):
        return self.positive_characteristics[random.randint(0, len(self.positive_characteristics) - 1)]
    
    def get_negative(self):
        return self.negative_characteristics[random.randint(0, len(self.negative_characteristics) - 1)]
    
    def get_characteristic(self, typ: int):
        if typ == self.POSITIVE:
            return self.get_positive()
        elif typ == self.NEGATIVE:
            return self.get_negative()
        raise NotImplementedError(typ)

    def create_characteristic_scene(self):
        """
        1: 0.6
        2: 0.3
        3: 0.1

        negative characteristic: 0.1
        """
        _rand = random.uniform(0, 1)
        if _rand < 0.6:
            chs = [(0, self.max_turns, self.POSITIVE)]
        elif _rand < 0.9:
            chs = []
            for _ in range(2):
                if random.uniform(0, 1) < 0.1:
                    # negative
                    ch = self.NEGATIVE
                else:
                    ch = self.POSITIVE
                chs.append(ch)
            if sum(chs) == 2:
                if random.uniform(0, 1) < 0.5:
                    chs[0] = self.POSITIVE
                else:
                    chs[1] = self.POSITIVE
            if sum == 0:
                return [(0, self.max_turns, self.POSITIVE)]
            if chs[0] == self.POSITIVE:
                chs = [
                    (0, self.max_turns - 3, self.POSITIVE),
                    (self.max_turns - 3, self.max_turns, self.NEGATIVE),
                ]
            else:
                chs = [
                    (0, self.max_turns - 3, self.NEGATIVE),
                    (self.max_turns - 3, self.max_turns, self.POSITIVE),
                ]
        else:
            p = random.randint(0, self.max_turns - 3 - 1)
            chs = [
                (0, p, self.POSITIVE),
                (p, p + 3, self.NEGATIVE),
                (p + 3, self.max_turns, self.POSITIVE),
            ]
            if chs[0][1] - chs[0][0] <= 0:
                chs = chs[1:]
            if chs[-1][1] - chs[-1][0] <= 0:
                chs = chs[:-1]

        turns = []
        for s, e, p in chs:
            for _ in range(s, e):
                turns.append(p)
        return turns


def startswith(text: str, starts: list[str]):
    for st in starts:
        if text.startswith(st):
            return st
    return None


class ScenePreparation(PreparationMixin):
    """
    Each element contains following attribute:
    - aspect: str
    - topic: str
    - scene: dict
        - role: list[str]
        - stages: list[str]
    """

    prompt_en = """Create a dialogue framework for a conversation based on the topic: {topic}. 
The framework should include:
1. Two distinct participant roles, each with unique perspectives, responsibilities, or expertise related to the topic. Do not use proper names for the characters.
2. A structure divided into three stages of the conversation:
- **Stage 1: Opening and setting the context** – Introduce the topic, establish the purpose, and outline each participant's role.
- **Stage 2: Core discussion or interaction** – Engage in an in-depth exchange of ideas, addressing challenges, exploring solutions, or making decisions.
- **Stage 3: Conclusion and follow-up** – Summarize key points, agree on outcomes or next steps, and provide final thoughts.

Each stage should include prompts or guiding behaviours to drive the conversation. The roles and stages should fit naturally within the topic and encourage dynamic and constructive interaction.

### Return output as follows:

### Roles:
- Role 1: <brief description for role 1 here>
- Role 2: <brief description for role 2 here>

### Stages:
- Stage 1: <description of behaviours of role 1 and role 2>
- Stage 2: <description of behaviours of role 1 and role 2>
- Stage 3: <description of behaviours of role 1 and role 2>
"""

    prompt_vi = """Tạo khung đối thoại cho một cuộc thảo luận dựa trên chủ đề: {topic}.
Khung đối thoại cần bao gồm:
1. Hai vai diễn tham gia riêng biệt, mỗi vai diễn có góc nhìn, trách nhiệm, hoặc chuyên môn riêng biệt liên quan đến chủ đề. Không được dùng tên riêng cho các vai diễn.
2. Cấu trúc được chia thành ba giai đoạn của cuộc thảo luận:
- **Giai đoạn 1: Mở đầu và thiết lập bối cảnh** – Giới thiệu chủ đề, xác định mục đích, và phác thảo vai trò của từng người tham gia.
- **Giai đoạn 2: Thảo luận chính hoặc tương tác** – Thực hiện trao đổi ý tưởng sâu sắc, giải quyết các thách thức, tìm kiếm giải pháp, hoặc đưa ra quyết định.
- **Giai đoạn 3: Kết luận và theo dõi** – Tóm tắt các điểm chính, thống nhất kết quả hoặc các bước tiếp theo, và đưa ra ý kiến cuối cùng.

Mỗi giai đoạn nên bao gồm các gợi ý hoặc hướng dẫn về hành vi để thúc đẩy cuộc trò chuyện. Vai diễn và các giai đoạn cần phù hợp và tự nhiên với chủ đề và khuyến khích sự tương tác năng động, mang tính xây dựng.

### Trả về kết quả như sau:

### Các vai diễn:
- Vai diễn 1: <tên vai diễn 1 tại đây, không có mô tả>
- Vai diễn 2: <tên vai diễn 2 tại đây, không có mô tả>

### Các giai đoạn:
- Giai đoạn 1: <mô tả ngắn gọn trong 1 câu các hành động của <vai diễn 1> và <vai diễn 2>>
- Giai đoạn 2: <mô tả ngắn gọn trong 1 câu các hành động của <vai diễn 1> và <vai diễn 2>>
- Giai đoạn 3: <mô tả ngắn gọn trong 1 câu các hành động của <vai diễn 1> và <vai diễn 2>>
"""

    def __init__(self, filepath, model_name = "gemini", enable_progress=False, auto_save=True):
        super().__init__(filepath, ["aspect", "topic"], model_name, enable_progress, False, auto_save)

    def get_prompt(self, topic: str, aspect: str=None):
        return self.prompt_vi.format(topic=topic)
    
    def parse_output(self, output: str, aspect: str, topic: str):
        topic is not None
        lines = output.strip().split("\n")
        is_role = False
        is_stages = False
        scene = {}

        for line in lines:
            line = line.strip()
            if line == "":
                continue

            if line.startswith("### Các vai diễn"):
                is_role = True
                scene["role"] = []

            elif startswith(line, ["- Vai diễn 1:", "- **Vai diễn 1:**", "- **Vai diễn 1**:"]):
                if is_role is False or len(scene["role"]) != 0:
                    return None
                prefix = startswith(line, ["- Vai diễn 1:", "- **Vai diễn 1:**", "- **Vai diễn 1**:"])
                scene["role"].append(line[len(prefix) :].strip())

            elif startswith(line, ["- Vai diễn 2:", "- **Vai diễn 2:**", "- **Vai diễn 2**:"]):
                if is_role is False or len(scene["role"]) != 1:
                    return None
                prefix = startswith(line, ["- Vai diễn 2:", "- **Vai diễn 2:**", "- **Vai diễn 2**:"])
                scene["role"].append(line[len(prefix) :].strip())

            elif line.startswith("### Các giai đoạn"):
                if is_role is False:
                    return None
                is_role = False
                is_stages = True
                scene["stages"] = []

            elif startswith(line, ["- Giai đoạn 1:", "- **Giai đoạn 1:**", "- **Giai đoạn 1**:"]):
                if is_stages is False or len(scene["stages"]) != 0:
                    return None
                prefix = startswith(line, ["- Giai đoạn 1:", "- **Giai đoạn 1:**", "- **Giai đoạn 1**:"])
                scene["stages"].append(line[len(prefix) :].strip())
            
            elif startswith(line, ["- Giai đoạn 2:", "- **Giai đoạn 2:**", "- **Giai đoạn 2**:"]):
                if is_stages is False or len(scene["stages"]) != 1:
                    return None
                prefix = startswith(line, ["- Giai đoạn 2:", "- **Giai đoạn 2:**", "- **Giai đoạn 2**:"])
                scene["stages"].append(line[len(prefix) :].strip())

            elif startswith(line, ["- Giai đoạn 3:", "- **Giai đoạn 3:**", "- **Giai đoạn 3**:"]):
                if is_stages is False or len(scene["stages"]) != 2:
                    return None
                prefix = startswith(line, ["- Giai đoạn 3:", "- **Giai đoạn 3:**", "- **Giai đoạn 3**:"])
                scene["stages"].append(line[len(prefix) :].strip())

        if len(scene["role"]) != 2:
            return None
        if len(scene["stages"]) != 3:
            return None
        
        # for stage in scene["stages"]:
        #     if "vai trò" in stage:
        #         return None
        
        return [{
            "aspect": aspect,
            "topic": topic,
            "scene": scene
        }]
    

class ConversationPreparation(PreparationMixin):
    """
    Xây dựng kịch bản gồm nhiều giai đoạn, dựa trên tính cách của người nói.
    Trong cuộc hội thoại có 2 người tham gia (User và Assistant), mỗi người sẽ có một kịch bản riêng.

    Chú ý: 
    - Assistant không được có thái độ tiêu cực
    - Có thể chuyển đổi từ thái độ <tích-cực-1> sang <tích-cực-2>
    - Không cần mở đầu, chào hỏi hay kết luận
    - Thái độ tiêu cực không được quá `3 turns`

    Xử lý:
    - current_user_characteristic
    - current_assistant_characteristic

    - Với mỗi turn: 
        - Cập nhật tính cách User và Assitant theo kịch bản
            - Xem có phải đổi tính cách không
            - Đối diện với User tiêu cực như thế nào ( + `deal_with_neg`)
    """

    ACTOR_USER = "actor_user"
    ACTOR_ASSISTANT = "actor_assistant"

    prompt = """{characteristic} Vai trò của bạn là {role}. Giới tính của bạn là {sex}. Giới tính người thảo luận với bạn là {partner_sex}.
    
Hãy thảo luận theo các giai đoạn sau:
{stages}.

Hãy thảo luận thật tự nhiên và sáng tạo, tránh gượng ép. Nhớ mỗi lần trả lời chỉ sử dụng 1 đến 2 câu. Chỉ được sử dụng tiếng Việt. Nhớ xưng hô thật thân thiệt, ví dụ như "cậu - tớ", "bạn - tớ", "bạn - mình", "ông - tôi", "bà - tôi"."""

    def __init__(self, filepath, model_name = "gemini", enable_progress=False, auto_save=True):
        super().__init__(filepath, ["aspect", "topic"], model_name, enable_progress, use_jsonl=False, auto_save=auto_save)

        self.max_turns = 10
        self.characteristic_gen = CharacteristicPreparation(self.max_turns)

    def get_prompt(self, id_turn: int, characteristic: str, sex: str, partner_sex: str, role: str, stages: list[str], is_partner_negative=False):
        stages = [
            f"- Giai đoạn 1: 4 câu đầu tiên: {stages[0]}",
            f"- Giai đoạn 2: 4 câu tiếp theo: {stages[1]}",
            f"- Giai đoạn 3: 2 câu cuối: {stages[2]}"
        ]
        if id_turn < 4:
            current_stage = stages[0]
        elif id_turn < 8:
            current_stage = stages[1]
        else:
            current_stage = stages[2]
        current_stage = current_stage[2:]
        prompt = self.prompt.format(
            characteristic=characteristic, 
            role=role, 
            stages="\n".join(stages), 
            # current_stage=current_stage, 
            sex=sex, 
            partner_sex=partner_sex
        )

        if is_partner_negative:
            prompt += "\n" + self.characteristic_gen.deal_with_neg

        return prompt

    def convert_messages_by_role(self, role: str, messages: list[str]):
        assert role in [self.ACTOR_USER, self.ACTOR_ASSISTANT]
        messages = copy.deepcopy(messages)
        new_messages = []
        for m in messages:
            if m["role"] == "system":
                new_messages.append(m)
            elif m["role"] == role:
                new_messages.append({"role": "assistant", "content": m["content"]})
            else:
                new_messages.append({"role": "user", "content": m["content"]})
        return new_messages

    def prepare_messages(self, role: str, system_prompt: str, current_stage_id: int, messages: list[str]):
        openai_messages = self.convert_messages_by_role(role, messages)
        openai_messages.insert(0, {"role": "system", "content": system_prompt})
        if len(openai_messages) == 1:
            openai_messages.append({"role": "user", "content": "Xin chào."})
        openai_messages[-1]["content"] += f"\n### Nhớ rằng bạn đang ở giai đoạn {current_stage_id}, mỗi lần trả lời chỉ sử dụng 1 đến 2 câu. Chỉ được sử dụng tiếng Việt."
        return openai_messages

    def call_messages(self, role: str, system_prompt: str, current_stage_id: int, messages: list[str]):
        openai_messages = self.prepare_messages(role, system_prompt, current_stage_id, messages)
        response = chat_completions(self.model_name)(
            n=1,
            messages=openai_messages,
            temperature=random.uniform(0.6, 1.2),
            max_tokens=2048,
        )
        content = response.choices[0].message.content
        messages.append({"role": role, "content": content})
        return messages

    def run(self, aspect: str, topic: str, scene: dict):
        messages = []

        ids = [0, 1]
        random.shuffle(ids)

        assistant_role = scene["role"][ids[1]]
        assistant_characteristic = self.characteristic_gen.get_positive()
        assistant_sex = random.choice(["nam", "nữ"])

        user_role = scene["role"][ids[0]]
        user_chars_scene = self.characteristic_gen.create_characteristic_scene()
        user_chars_type = None
        user_characteristic = None
        user_sex = random.choice(["nam", "nữ"])
        user_characteristic_list = []

        for id_turn in range(self.max_turns):
            if user_chars_type != user_chars_scene[id_turn]:
                user_chars_type = user_chars_scene[id_turn]
                user_characteristic = self.characteristic_gen.get_characteristic(user_chars_type)
            user_characteristic_list.append(user_characteristic)
            
            if id_turn < 4:
                stage_id = 1
            elif id_turn < 8:
                stage_id = 2
            else:
                stage_id = 3

            user_system_prompt = self.get_prompt(
                id_turn, 
                characteristic=user_characteristic, 
                sex=user_sex, 
                partner_sex=assistant_sex, 
                role=user_role, 
                stages=scene["stages"],
            )
            assistant_system_prompt = self.get_prompt(
                id_turn, 
                characteristic=assistant_characteristic, 
                sex=assistant_sex, 
                partner_sex=user_sex, 
                role=assistant_role, 
                stages=scene["stages"],
            )

            messages = self.call_messages(self.ACTOR_USER, user_system_prompt, stage_id, messages)
            messages = self.call_messages(self.ACTOR_ASSISTANT, assistant_system_prompt, stage_id, messages)

        self.add_values({
            "aspect": aspect,
            "topic": topic,
            "scene": scene,
            "metadata": {
                "sex": {
                    "user": user_sex,
                    "assistant": assistant_sex,
                },
                "role": {
                    "user": user_role,
                    "assistant": assistant_role
                },
                "user_characteristic": {i: ch for i, ch in enumerate(user_characteristic_list)}
            },
            "messages": messages,
        })

        self.save()
    
    def prepare_seed(self, scene: dict):
        ids = [0, 1]
        random.shuffle(ids)

        assistant_role = scene["role"][ids[1]]
        assistant_characteristic = self.characteristic_gen.get_positive()
        assistant_sex = random.choice(["nam", "nữ"])

        user_role = scene["role"][ids[0]]
        user_chars_scene = self.characteristic_gen.create_characteristic_scene()
        user_chars_type = None
        user_characteristic = None
        user_sex = random.choice(["nam", "nữ"])

        user_characteristic_list = []

        for id_turn in range(self.max_turns):
            if user_chars_type != user_chars_scene[id_turn]:
                user_chars_type = user_chars_scene[id_turn]
                user_characteristic = self.characteristic_gen.get_characteristic(user_chars_type)
            user_characteristic_list.append(user_characteristic)

        return dict(
            assistant_role=assistant_role,
            assistant_characteristic=assistant_characteristic,
            assistant_sex=assistant_sex,

            user_role=user_role,
            user_characteristic_list=user_characteristic_list,
            user_sex=user_sex,

            stages=scene["stages"],
        )
    
    def prepare_single_turn(self, id_turn: int, messages: list[dict], seed: dict, id: int=None):
        """
        Parameters
        ----------
            id_turn: int
            messages: list[dict]
            seed: dict
                Check function `prepare_seed()`
        """
        if id_turn < 4:
            stage_id = 1
        elif id_turn < 8:
            stage_id = 2
        else:
            stage_id = 3
        
        if len(messages) == 0:
            last = {"role": self.ACTOR_ASSISTANT}
        else:
            last = messages[-1]
        
        if last["role"] == self.ACTOR_ASSISTANT:
            # TODO. Now, it is user's turn
            user_system_prompt = self.get_prompt(
                id_turn, 
                characteristic=seed["user_characteristic_list"][id_turn], 
                sex=seed["user_sex"], 
                partner_sex=seed["assistant_sex"], 
                role=seed["user_role"], 
                stages=seed["stages"],
            )
            if id is not None:
                user_system_prompt = str(id) + "\n" + user_system_prompt
            messages = self.prepare_messages(self.ACTOR_USER, user_system_prompt, stage_id, messages)

        elif last["role"] == self.ACTOR_USER:
            # TODO. Now, it is assistant's turn
            is_user_negative = self.characteristic_gen.status(["user_characteristic_list"][id_turn]) # User always starts first
            is_user_negative = bool(is_user_negative == self.characteristic_gen.NEGATIVE)
            assistant_system_prompt = self.get_prompt(
                id_turn, 
                characteristic=seed["assistant_characteristic"], 
                sex=seed["assistant_sex"], 
                partner_sex=seed["user_sex"], 
                role=seed["assistant_role"], 
                stages=seed["stages"],
                is_partner_negative=is_user_negative,
            )
            if id is not None:
                assistant_system_prompt = str(id) + "\n" + assistant_system_prompt
            messages = self.prepare_messages(self.ACTOR_ASSISTANT, assistant_system_prompt, stage_id, messages)
        
        return convert_openai_messages_to_gemini_request(messages)


def convert_openai_messages_to_gemini_request(messages):
    item = {
        "system_instruction": None,
        "contents": [],
        "generationConfig": {
            "temperature": random.uniform(0.6, 1.2),
            "max_output_tokens": 2048
        }
    }
    for msg in messages:
        if msg["role"] == "system":
            item["system_instruction"] = {
                "parts": [
                    {"text": msg["content"]}
                ]
            }
        elif msg["role"] == "user":
            item["contents"].append({
                "role": "user",
                "parts": [
                    {"text": msg["content"]}
                ]
            })
        elif msg["role"] == "assistant":
            item["contents"].append({
                "role": "model",
                "parts": [
                    {"text": msg["content"]}
                ]
            })
        else:
            raise ValueError(msg)

    if item["system_instruction"] is None:
        item.pop("system_instruction")
    return {"request": item}
        

def main(step: int):
    aspect_file = "data/long_conversation/00_aspects.jsonl"
    topic_file = "data/long_conversation/01_topics.jsonl"
    scene_file = "data/long_conversation/02_scenes.json"
    conversation_file = "data/long_conversation/03_conversations.json"

    aspect_gen = AspectPreparation(aspect_file, enable_progress=True)
    if step <= 0:
        aspect_gen.run(50)

    topic_gen = TopicPreparation(topic_file, enable_progress=True)
    if step <= 1:
        for aspect in tqdm(aspect_gen.values, desc="Topic preparation"):
            aspect = aspect["aspect"]
            for prompt_ver in range(len(TopicPreparation.prompt_list)):
                topic_gen.run(1, aspect=aspect, prompt_version=prompt_ver)

    scene_gen = ScenePreparation(scene_file)
    conversation_gen = ConversationPreparation(conversation_file)
    if step <= 2:
        for topic in tqdm(topic_gen.values[:], desc="Scene preparation"):
            scenes = scene_gen.run(1, aspect=topic["aspect"], topic=topic["topic"] + " - " + topic["description"])

            for scene in scenes:
                for _ in range(3):
                    conversation_gen.run(scene["aspect"], scene["topic"], scene["scene"])


def sort_dict_key(value):
    if isinstance(value, dict):
        return {k: sort_dict_key(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [sort_dict_key(v) for v in value]
    if isinstance(value, tuple):
        return tuple([sort_dict_key(v) for v in value])
    if isinstance(value, set):
        return set([sort_dict_key(v) for v in value])
    return value


def encode_dict(_dict):
    return hashlib.sha256(str(sort_dict_key(_dict)).encode()).hexdigest()


def main_batch(step: int):
    """
    Note that `aspect` and `topic` must be completely prepared
    """
    topic_file = "data/long_conversation/01_topics.jsonl"
    scene_requests_file = "data/long_conversation/12_scenes_requests.jsonl"
    scene_full_requests_file = "data/long_conversation/12_scenes_requests_full.jsonl"
    scene_responses_file = "data/long_conversation/12_scenes_responses.jsonl"
    conversation_seeds_file = "data/long_conversation/13_conversation_seeds.jsonl"

    topic_gen = TopicPreparation(topic_file)

    scene_gen = ScenePreparation(None)
    if step == 0:
        _scene_requests = []
        for __topic in topic_gen.values:
            __kwargs = dict(aspect=__topic["aspect"], topic=__topic["topic"] + " - " + __topic["description"])
            __request = scene_gen.prepare_input_requests(5, **__kwargs)
            _scene_requests.extend({
                **__kwargs,
                "request": __request
            })
        with jsonlines.open(scene_requests_file, "w") as f:
            f.write_all([req["request"] for req in _scene_requests])
        with jsonlines.open(scene_full_requests_file, "w") as f:
            f.write_all(_scene_requests)
    
    if step == 1:
        with jsonlines.open(scene_full_requests_file) as f:
            _scene_full_requests_dict = {}
            for __item in f:
                _scene_full_requests_dict[encode_dict(__item["request"])] = {k: v for k, v in __item.items() if k!= "request"}

        with jsonlines.open(scene_responses_file) as f:
            _scene_responses = list(f)

        assert len(_scene_full_requests_dict) == len(_scene_responses), f"{len(_scene_full_requests_dict)} != {len(_scene_responses)}"

        _successes = []
        _errors = []
        for __response in _scene_responses:
            __additional_inputs = _scene_full_requests_dict[encode_dict(__response["request"])]
            __output = __response["response"]["candidates"][0]["content"]["parts"][0]["text"]
            __finish_reason = __response["response"]["candidates"][0]["finishReason"]
            if __finish_reason == "STOP":
                __scene = scene_gen.parse_output(__output, **__additional_inputs)
                _successes.append(__scene[0])
            else:
                _errors.append(__response)
        print("Number of errors:", len(_errors))
        
        # TODO. Prepare seed for conversation
        _conversation_gen = ConversationPreparation(None)
        _conversation_seeds = []
        for __scene in _successes:
            for _ in range(5):
                ___seed = _conversation_gen.prepare_seed(__scene["scene"])
                _conversation_seeds.append({
                    **{k: v for k, v in __scene.items() if k != "scene"},
                    **___seed
                })
        with jsonlines.open(conversation_seeds_file, "w") as f:
            f.write_all(_conversation_seeds)

    # TODO. Start the conversation
    if step == 2:
        with jsonlines.open(conversation_seeds_file) as f:
            _conversation_seeds = list(f)

        _conversation_gen = ConversationPreparation(None)
        _id_turn = 0
        _requests = []
        for __seed in _conversation_seeds:
            __request = _conversation_gen.prepare_single_turn(_id_turn, [], __seed)
            _requests.append({
                **__seed,
                "request": __request,
            })

        conversation_requests_file = f"data/long_conversation/14_conversations_requests_turn_{_id_turn}.jsonl"
        conversation_full_requests_file = f"data/long_conversation/14_conversations_full_requests_turn_{_id_turn}.jsonl"

        with jsonlines.open(conversation_full_requests_file, "w") as f:
            f.write_all(_requests)
        with jsonlines.open(conversation_requests_file, "w") as f:
            f.write_all([r["request"] for r in _requests])

    # TODO. Continue the conversation
    # if step <= 3:
    #     with jsonlines.open(conversation_seeds_file) as f:
    #         conversation_seeds = list(f)


if __name__ == "__main__":
    main(0)
    # main_batch(2)