import os
import re
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import anthropic


class ExtractError(Exception):
    pass


class EvaluationMixin:
    LEFT = 0
    RIGHT = 1
    EQUAL = 2
    
    def __init__(self, client: anthropic.Anthropic=None):
        if client is None:
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        else:
            self.client = client
        
    def call(self, prompt: str, system_prompt: str=None) -> str:
        kwargs = dict(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.6,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
            
        message = self.client.messages.create(**kwargs)
        content = message.content[0].text
        return content
    
    def evaluate(self, left, right, shared):
        raise NotImplementedError()
    
    @staticmethod
    def findall(text: str, sub: str) -> list[int]:
        pos_ids = []
        start = 0
        while True:
            pos = text.find(sub, start)
            if pos == -1:
                break
            pos_ids.append(pos)
            start = pos + len(sub)
        return pos_ids
    
    @classmethod
    def get_output_by_markers(cls, text: str, start_marker: str, end_marker: str):
        pos_ids = cls.findall(text, start_marker)   
        if len(pos_ids) != 1:
            return
        start = pos_ids[0]
        
        pos_ids = cls.findall(text, end_marker)    
        if len(pos_ids) != 1:
            return
        end = pos_ids[0]
        
        return text[start + len(start_marker) : end]
    
    @staticmethod
    def extract_score(output: str, criterion: str):
        output = output.lower()
        criterion = criterion.lower()
        patterns = [
            rf"{criterion}: (\d+)",
            rf"\*\*{criterion}:\*\* (\d+)",
            rf"\*\*{criterion}\*\*: (\d+)",
            
            rf"{criterion}: \*\*(\d+)\*\*",
            rf"\*\*{criterion}:\*\* \*\*(\d+)\*\*",
            rf"\*\*{criterion}\*\*: \*\*(\d+)\*\*",

            rf"- {criterion}: (\d+)",
            rf"- {criterion}: \*\*(\d+)\*\*",
            rf"- {criterion}: \*\*(\d+)\*\*",
            
            rf"- \*\*{criterion}:\*\* (\d+)",
            rf"- \*\*{criterion}:\*\* \*\*(\d+)\*\*",
            rf"- \*\*{criterion}:\*\* \*\*(\d+)\*\*",
        ]
        for pattern in patterns:
            matches = re.search(pattern, output, re.DOTALL)
            if matches:
                try:
                    return int(matches.group(1))
                except ValueError:
                    return float(matches.group(1))
        # raise ExtractError("Cannot extract score from output")
    
    @staticmethod
    def extract_value(output: str, criterion: str):
        output = output.lower()
        criterion = criterion.lower()
        patterns = [
            rf"{criterion}: (\w+)",
            rf"\*\*{criterion}:\*\* (\w+)",
            rf"\*\*{criterion}\*\*: (\w+)",
            
            rf"{criterion}: \*\*(\w+)\*\*",
            rf"\*\*{criterion}:\*\* \*\*(\w+)\*\*",
            rf"\*\*{criterion}\*\*: \*\*(\w+)\*\*",

            rf"- {criterion}: (\w+)",
            rf"- {criterion}: \*\*(\w+)\*\*",
            rf"- {criterion}: \*\*(\w+)\*\*",
            
            rf"- \*\*{criterion}:\*\* (\w+)",
            rf"- \*\*{criterion}:\*\* \*\*(\w+)\*\*",
            rf"- \*\*{criterion}:\*\* \*\*(\w+)\*\*",
        ]
        for pattern in patterns:
            matches = re.search(pattern, output, re.DOTALL)
            if matches:
                return matches.group(1)
        raise ExtractError("Cannot extract score from output")

    def run_with_retry(self, num_retrials: int, func, *args, **kwargs):
        trials = 1
        while trials <= num_retrials:
            trials += 1
            
            ret = func(*args, **kwargs)
            if ret is not None:
                return ret
            
            if trials <= num_retrials:
                print(f"{self.__class__.__name__}: Retry {trials} / {num_retrials}")


class TranslationEvaluation(EvaluationMixin):
    CORRECTNESS_PROMPT = """You are an expert language translator and evaluator. Your task is to assess the quality of a given translation from one language to another. Please carefully review the following information:

<original_text>
{original_text}
</original_text>

<translation>
{translation}
</translation>

<source_language>
{source_language}
</source_language>

<target_language>
{target_language}
</target_language>

Your evaluation should focus on four key aspects:
1. Accuracy: Does the translation convey the same meaning as the original text?
2. Completeness: Has any information been omitted or added?
3. Grammar and syntax: Is the translation grammatically correct in the target language?
4. Style and tone: Does the translation maintain the appropriate style and tone of the original?

Please conduct your evaluation using the following steps:

1. In <evaluation_process> tags:
   a. Write down key phrases or sentences from both the original text and translation, aligned side by side. This will help in comparing accuracy and completeness. It's OK for this section to be quite long.
   
   b. For each aspect of the translation (Accuracy, Completeness, Grammar and syntax, Style and tone):
      - Consider arguments for both strengths and weaknesses
      - Provide specific examples from the text to support your assessment
      - Assign a rating (Excellent, Good, Fair, or Poor) based on your analysis

2. After your evaluation process, provide your final decision and overall rating in <output> tags:
   - Decision: "correct" if the translation accurately conveys the original meaning without significant errors, or "incorrect" if there are substantial mistakes, omissions, or alterations in meaning.
   - Overall Rating: Assign an overall rating (Excellent, Good, Fair, or Poor) based on your comprehensive evaluation.

Example output structure (do not copy this content, it's just to illustrate the format):

<evaluation_process>
Key phrases comparison:
Original: [phrase 1]
Translation: [corresponding translation]

Original: [phrase 2]
Translation: [corresponding translation]

...

1. Accuracy:
   Strengths: [Your analysis]
   Weaknesses: [Your analysis]
   Examples: [Specific examples]
   Rating: [Your rating]

2. Completeness:
   Strengths: [Your analysis]
   Weaknesses: [Your analysis]
   Examples: [Specific examples]
   Rating: [Your rating]

3. Grammar and syntax:
   Strengths: [Your analysis]
   Weaknesses: [Your analysis]
   Examples: [Specific examples]
   Rating: [Your rating]

4. Style and tone:
   Strengths: [Your analysis]
   Weaknesses: [Your analysis]
   Examples: [Specific examples]
   Rating: [Your rating]
</evaluation_process>

<output>
Decision: [correct/incorrect]
Overall Rating: [Excellent/Good/Fair/Poor]
</output>

Please proceed with your evaluation of the provided translation.
"""

    COMPARISON_PROMPT = """You are an expert language translator and evaluator. Your task is to analyze and compare two translations of a given text, then choose the better version. The original text is in {source_language}, and both translations are in {target_language}.

Here are the texts you need to evaluate:

Original text ({source_language}):
<original_text>
{original_text}
</original_text>

Translation 1 ({target_language}):
<translation_1>
{translation_1}
</translation_1>

Translation 2 ({target_language}):
<translation_2>
{translation_2}
</translation_2>

Instructions:
1. Carefully read the original text and both translations.
2. Analyze each translation for accuracy, fluency, and style.
3. Compare the two translations, considering factors such as:
   - Accuracy in conveying the original meaning
   - Natural flow and readability in the target language
   - Appropriate use of idioms and cultural context
   - Consistency in tone and style with the original
4. Determine which translation is better overall.

Wrap your evaluation in <translation_evaluation> tags. Follow these steps in your analysis:

1. Provide a brief overview of the original text's main ideas and tone.
2. Create a side-by-side comparison of specific phrases or sentences from both translations, highlighting differences.
3. List pros and cons for each translation.
4. Discuss how well each translation captures the nuances of the original text.
5. Consider the overall readability and impact of each translation for the target audience.

After your analysis, provide your decision in the following format:
<decision>
Better translation: [Enter 1 or 2]
</decision>

Example output structure:

<translation_evaluation>
[Your detailed evaluation following the steps outlined above]
</translation_evaluation>

<decision>
Better translation: [1 or 2]
</decision>

Please proceed with your evaluation and decision.
"""

    rating_to_score = {
        "poor": 0,
        "fair": 1, 
        "good": 2, 
        "excellent": 3,
    }
    decision_to_score = {
        "incorrect": 0,
        "correct": 1
    }
    
    def __init__(self, src_lang: str, tgt_lang: str, client = None):
        super().__init__(client)
        
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def evaluate(self, left: str, right: str, original_text: str):
        left_correctness = self.evaluate_correctness(left, original_text)
        if left_correctness is None:
            return
        right_correctness = self.evaluate_correctness(right, original_text)
        if right_correctness is None:
            return
        
        if self.decision_to_score(left_correctness["decision"]) < self.decision_to_score(right_correctness["decision"]):
            return self.RIGHT
        if self.decision_to_score(left_correctness["decision"]) > self.decision_to_score(right_correctness["decision"]):
            return self.LEFT
        
        if self.rating_to_score(left_correctness["rating"]) < self.rating_to_score(right_correctness["rating"]):
            return self.RIGHT
        if self.rating_to_score(left_correctness["rating"]) > self.rating_to_score(right_correctness["rating"]):
            return self.LEFT
        
        comparison = self.evaluate_comparison(left, right, original_text)
        if comparison is None:
            return
        
        if comparison["better"] == 1:
            return self.LEFT
        if comparison["better"] == 2:
            return self.RIGHT
        return self.EQUAL

    def evaluate_correctness(self, translation: str, original_text: str):
        return self.run_with_retry(3, self._evaluate_correctness, translation, original_text)
        
    def _evaluate_correctness(self, translation: str, original_text: str):
        prompt = self.CORRECTNESS_PROMPT.format(original_text=original_text, translation=translation, source_language=self.src_lang, target_language=self.tgt_lang)
        response = self.call(prompt=prompt)
        
        output = self.get_output_by_markers(response, "<output>", "</output>")
        if output is None:
            return
        
        output = output.strip()
        decision = self.extract_value(output, "Decision").lower()
        rating = self.extract_value(output, "Overall Rating").lower()
        if decision not in ["correct", "incorrect"]:
            return
        if rating not in self.rating_to_score:
            return
        
        return {"decision": decision, "rating": rating}

    def evaluate_comparison(self, left: str, right: str, original_text: str):
        return self.run_with_retry(3, self._evaluate_comparison, left, right, original_text)
        
    def _evaluate_comparison(self, left: str, right: str, original_text: str):
        prompt = self.COMPARISON_PROMPT.format(original_text=original_text, source_language=self.src_lang, target_language=self.tgt_lang, translation_1=left, translation_2=right)
        response = self.call(prompt=prompt)
        
        output = self.get_output_by_markers(response, "<decision>", "</decision>")
        if output is None:
            return
        
        ret = self.extract_score(output, "Better translation")
        if ret not in [1, 2, 1.0, 2.0]:
            return
        
        return {"better": int(ret)}
        

class VietnameseCultureAndHistoryEvaluation(EvaluationMixin):
    HISTORICAL_CORRECTNESS_AND_COMPARISON_PROMPT = """You are a professional Vietnamese historian tasked with evaluating two answers to a given historical question. Your evaluation will assess the correctness and quality of each answer, and determine which one is better. All content (question and answers) will be in Vietnamese.

Here is the historical question:

<historical_question>
{question}
</historical_question>

Here is the first answer to evaluate:

<answer_one>
{answer_1}
</answer_one>

Here is the second answer to evaluate:

<answer_two>
{answer_2}
</answer_two>

Instructions:
1. Carefully read the historical question and both answers.
2. Analyze each answer separately, considering the following:
   - Historical accuracy
   - Relevance to the question
   - Depth of analysis
   - Use of evidence
   - Clarity of expression
3. Determine whether each answer is correct or incorrect.
4. Rate the quality of each answer using this scale:
   - Excellent: Comprehensive, insightful, well-supported
   - Good: Accurate and relevant, but may lack depth or detail
   - Fair: Generally on-topic, but with significant omissions or inaccuracies
   - Poor: Largely irrelevant, inaccurate, or lacking substance
5. Compare the two answers and decide which one is better overall.

Before providing your final evaluation, show your thought process for each answer inside <evaluation_process> tags. In this section:
- Quote relevant parts of each answer that support your evaluation.
- Consider arguments for and against each answer's correctness and quality.
- List out the criteria for evaluation (historical accuracy, relevance, depth, evidence, clarity) and rate each answer on these criteria individually.
- Summarize your findings before making a final decision.

This will ensure a thorough and transparent evaluation. It's OK for this section to be quite long.

After your evaluation process, provide your final evaluation using the following format:

<output>
<answer_1>
Answer 1:
- Correctness: [correct/incorrect]
- Overall Rating: [Excellent/Good/Fair/Poor]
- Reasoning: [Brief explanation of your rating]
</answer_1>

<answer_2>
Answer 2:
- Correctness: [correct/incorrect]
- Overall Rating: [Excellent/Good/Fair/Poor]
- Reasoning: [Brief explanation of your rating]
</answer_2>

<comparison>
- Better answer: [1 or 2]
- Justification: [Brief explanation of why this answer is better]
</comparison>
</output>

Remember to maintain objectivity and base your evaluation solely on the historical merit of the answers. Your expertise as a Vietnamese historian is crucial in providing an accurate and fair assessment.
"""

    CULTURAL_CORRECTNESS_AND_COMPARISON_PROMPT = """You are a distinguished Vietnamese cultural professor tasked with evaluating two answers to a given cultural question. Your evaluation will assess the correctness and quality of each answer, and determine which one is superior. All content (question and answers) will be in Vietnamese.

Here is the cultural question you need to consider:

<cultural_question>
{question}
</cultural_question>

Now, let's examine the two answers provided:

Answer 1:
<answer_one>
{answer_1}
</answer_one>

Answer 2:
<answer_two>
{answer_2}
</answer_two>

Please follow these steps to evaluate the answers:

1. Carefully read the cultural question and both answers.
2. Analyze each answer separately, considering the following criteria:
   - Cultural accuracy
   - Relevance to the question
   - Depth of cultural insight
   - Use of cultural evidence
   - Clarity of expression
3. Determine whether each answer is correct or incorrect based on its cultural content.
4. Rate the quality of each answer using this scale:
   - Excellent: Comprehensive, insightful, well-supported with cultural evidence
   - Good: Accurate and relevant, but may lack depth or detailed cultural context
   - Fair: Generally on-topic, but with significant omissions or cultural inaccuracies
   - Poor: Largely irrelevant, culturally inaccurate, or lacking substance
5. Compare the two answers and decide which one is better overall from a cultural perspective.

Before providing your final evaluation, conduct your cultural evaluation inside <cultural_evaluation> tags:

1. For each answer:
   - Quote relevant cultural information and insights
   - List strengths from a cultural perspective
   - List weaknesses from a cultural perspective
2. Compare the two answers directly:
   - Note similarities in cultural content and approach
   - Highlight key differences in cultural understanding and presentation

This will ensure a thorough interpretation of the cultural content.

After your evaluation, present your final assessment in the following format:

<output>
<answer_1>
Answer 1:
- Correctness: [correct/incorrect]
- Overall Rating: [Excellent/Good/Fair/Poor]
- Reasoning: [Brief explanation of your rating, focusing on cultural aspects]
</answer_1>

<answer_2>
Answer 2:
- Correctness: [correct/incorrect]
- Overall Rating: [Excellent/Good/Fair/Poor]
- Reasoning: [Brief explanation of your rating, focusing on cultural aspects]
</answer_2>

<comparison>
- Better answer: [1 or 2]
- Justification: [Brief explanation of why this answer is culturally superior]
</comparison>
</output>

Please proceed with your evaluation of the answers to the cultural question.
"""

    def evaluate(self, left: str, right: str, question: str, type: str):
        return self.run_with_retry(3, self._evaluate, left, right, question, type)

    def _evaluate(self, left: str, right: str, question: str, type: str):
        assert type in ["history", "culture"]
        if type == "history":
            prompt = self.HISTORICAL_CORRECTNESS_AND_COMPARISON_PROMPT.format(question=question, answer_1=left, answer_2=right)
        else:
            prompt = self.CULTURAL_CORRECTNESS_AND_COMPARISON_PROMPT.format(question=question, answer_1=left, answer_2=right)

        response = self.call(prompt)
        
        output = self.get_output_by_markers(response, "<output>", "</output>")
        if output is None:
            return
        
        ret_ans_1 = self.get_output_by_markers(output, "<answer_1>", "</answer_1>")
        if ret_ans_1 is None:
            return
        ret_ans_2 = self.get_output_by_markers(output, "<answer_2>", "</answer_2>")
        if ret_ans_2 is None:
            return
        
        comparison = self.get_output_by_markers(output, "<comparison>", "</comparison>")
        if comparison is None:
            return
        
        metadata = {
            "left": {
                "correctness": self.extract_value(ret_ans_1, "Correctness"),
                "rating": self.extract_value(ret_ans_1, "Overall Rating"),
            },
            "right": {
                "correctness": self.extract_value(ret_ans_2, "Correctness"),
                "rating": self.extract_value(ret_ans_2, "Overall Rating"),
            }
        }
        
        comparison = self.extract_score(comparison, "Better answer")
        if comparison == 1:
            return self.LEFT, metadata
        if comparison == 2:
            return self.RIGHT, metadata
        return self.EQUAL, metadata
    

class PoemAndStoryTellingEvaluation(EvaluationMixin):
    POEM_PROMPT = """You are a professional poet tasked with evaluating and comparing two poems based on a given requirement. Your goal is to provide a thoughtful analysis of each poem and determine which one better meets the specified criteria. All content (requirement and poems) will be in Vietnamese.

First, carefully read the following requirement:

<requirement>
{requirement}
</requirement>

Now, read and analyze the two poems:

<poem_1>
{poem_1}
</poem_1>

<poem_2>
{poem_2}
</poem_2>

Before providing your final evaluation, break down your thought process in <poem_analysis> tags. For each poem, follow these steps:

1. Quote relevant lines that directly address the given requirement.
2. Analyze the poem's structure, rhythm, and use of poetic devices.
3. Evaluate the poem's imagery and emotional impact.
4. Note any unique or standout elements in the poem.
5. Assess how well the poem meets the given requirement overall.

After analyzing both poems individually, compare them directly based on how well they meet the requirement.

After your analysis, provide your final evaluation using the following format:

<output>
<poem_1_evaluation>
Poem 1:
- Overall Rating: [Excellent/Good/Fair/Poor]
- Strengths: [List 2-3 key strengths]
- Areas for Improvement: [List 1-2 areas, if applicable]
</poem_1_evaluation>

<poem_2_evaluation>
Poem 2:
- Overall Rating: [Excellent/Good/Fair/Poor]
- Strengths: [List 2-3 key strengths]
- Areas for Improvement: [List 1-2 areas, if applicable]
</poem_2_evaluation>

<comparison>
- Better poem: [1 or 2]
- Reason for selection: [Brief explanation of why this poem was chosen as better]
</comparison>
</output>

Remember to base your evaluation primarily on how well each poem meets the given requirement, while also considering overall poetic quality.
"""

    STORY_TELLING_PROMPT = """You are an experienced literary critic and storyteller tasked with evaluating two stories based on a given set of requirements. Your goal is to provide a thorough analysis of each story, rate them, and determine which one is better. All content (requirement and stories) will be in Vietnamese.

First, carefully read the requirement and both stories:

Requirement:
<requirement>
{requirement}
</requirement>

Story 1:
<story_1>
{story_1}
</story_1>

Story 2:
<story_2>
{story_2}
</story_2>

Now, please evaluate each story using the following process:

1. Analyze how well the story meets the given requirement.
2. Identify the story's strengths and weaknesses.
3. Assign an overall rating (Excellent/Good/Fair/Poor) based on your analysis.

Wrap your thought process for each story in <story_analysis> tags before providing the final output. Inside these tags:

1. Identify and quote key passages from the story that relate to the requirement.
2. List out specific elements of the story (plot, characters, setting, etc.) and how they contribute to meeting the requirement.
3. Provide a numbered list of strengths and weaknesses, with brief explanations for each.
4. Consider arguments for each possible overall rating (Excellent/Good/Fair/Poor) before deciding on the final rating.

After evaluating both stories, compare them and choose the better one. Provide a brief explanation for your choice.

Present your final analysis in the following format:

<output>
<story_1_evaluation>
Story 1:
- Overall Rating: [Excellent/Good/Fair/Poor]
- Strengths: [List key strengths]
- Weaknesses: [List key weaknesses]
- Requirement Fulfillment: [Explain how well the story meets the requirement]
</story_1_evaluation>

<story_2_evaluation>
Story 2:
- Overall Rating: [Excellent/Good/Fair/Poor]
- Strengths: [List key strengths]
- Weaknesses: [List key weaknesses]
- Requirement Fulfillment: [Explain how well the story meets the requirement]
</story_2_evaluation>

<comparison>
- Better story: [1 or 2]
- Explanation: [Briefly explain why this story is better]
</comparison>
</output>

Remember to provide thoughtful and constructive feedback for each story, focusing on specific elements that contribute to its quality and how well it meets the given requirement.
"""

    def evaluate(self, left, right, requirement, type: str):
        return self.run_with_retry(3, self._evaluate, left, right, requirement, type)

    def _evaluate(self, left, right, requirement, type: str):
        assert type in ["poem", "story"]
        if type == "poem":
            prompt = self.POEM_PROMPT.format(requirement=requirement, poem_1=left, poem_2=right)
        else:
            prompt = self.STORY_TELLING_PROMPT.format(requirement=requirement, story_1=left, story_2=right)
        
        response = self.call(prompt)

        output = self.get_output_by_markers(response, "<output>", "</output>")
        if output is None:
            return

        ret_ans_1 = self.get_output_by_markers(output, f"<{type}_1_evaluation>", f"</{type}_1_evaluation>")
        if ret_ans_1 is None:
            return
        ret_ans_2 = self.get_output_by_markers(output, f"<{type}_2_evaluation>", f"</{type}_2_evaluation>")
        if ret_ans_2 is None:
            return

        comparison = self.get_output_by_markers(output, "<comparison>", "</comparison>")
        if comparison is None:
            return

        metadata = {
            "left": {
                "rating": self.extract_value(ret_ans_1, "Overall Rating"),
            },
            "right": {
                "rating": self.extract_value(ret_ans_2, "Overall Rating"),
            }
        }

        comparison = self.extract_score(comparison, f"Better {type}")
        if comparison == 1:
            return self.LEFT, metadata
        if comparison == 2:
            return self.RIGHT, metadata
        return self.EQUAL, metadata


if __name__ == "__main__":
    evaluator = PoemAndStoryTellingEvaluation()

    print(evaluator.evaluate(
        requirement="Hãy sáng tác một bài thơ lấy cảm hứng từ chủ đề hoặc cụm từ sau đây: Biến đổi khí hậu và môi trường - Gọi mời hành động bảo vệ môi trường thông qua thơ ca.",
        right="Mẹ Trái Đất thở than, hơi nóng rát da\nDòng sông bạc xưa nay, nay chỉ còn phù sa\nMưa bão hung tàn đến, gió rét cắt da thịt\nCánh đồng vàng úa héo, tiếng chim rơi lặng thinh.\n\n\nRừng già rụng lá xanh, khói đen phủ mờ trời\nNúi non trơ sỏi đá, nguồn nước cạn khô rồi\nNhững con thuyền nan nhỏ, lênh đênh trên bão tố\nCâu hát ru buồn thiu, vọng mãi giữa mênh mông.\n\n\nBà già làng kể chuyện, thuở trước mùa màng tươi\nSuối mát róc rách chảy, chim muông gọi nhau chơi\nGió đưa hương lúa chín, thơm nồng khắp đồng quê\nNay chỉ còn khói bụi, bao trùm khắp mọi nơi.\n\n\nCon cháu ta mai sau, nhìn cảnh tượng này sao?\nĐất khô cằn nứt nẻ, biển cả dậy sóng cao\nChim bay lạc phương hướng, cá chết trôi lênh đênh\nLòng người sao cứ quên, trách nhiệm với quê hương?\n\n\nHãy thức tỉnh nào hỡi, những tâm hồn Việt Nam!\nBảo vệ rừng xanh ngát, giữ sạch dòng sông ngòi\nTrồng thêm cây xanh tốt, để cho đất thở dài\nHãy chung tay góp sức, cho đời thêm tươi sáng.\n\n\nTừ những hành động nhỏ, tình yêu sẽ lớn lên\nVì một tương lai xanh, cho con cháu mai sau\nHãy bảo vệ môi trường, đừng để trái đất đau\nTương lai tươi sáng mãi, Việt Nam đẹp muôn đời.\n",
        left="""Biến đổi khí hậu, thiên tai dâng lên,
Mưa rơi bất chợt, bão tố tơi bời,
Cây cối khô héo, biển xanh cũng vơi,
Trái đất khắc khoải, kêu gọi mỗi người.

Môi trường ô nhiễm, không khí ngột ngạt,
Rừng cây mất đi, cạn kiệt nguồn sống,
Chúng ta đừng đứng nhìn, đừng mãi thờ ơ,
Hành động ngay hôm nay, đừng để quá muộn màng.

Hãy chung tay bảo vệ, giữ gìn đất mẹ,
Từng hành động nhỏ, tạo ra sức mạnh lớn,
Trồng cây, giảm rác, tiết kiệm năng lượng,
Hãy cho thế hệ sau một hành tinh xanh.

Biến đổi khí hậu không phải điều xa vời,
Chúng ta có thể thay đổi, tạo nên sự khác biệt,
Bằng tình yêu và trách nhiệm với trái đất,
Hãy cùng nhau hành động, bảo vệ môi trường hôm nay!""",
    type="poem"
    ))