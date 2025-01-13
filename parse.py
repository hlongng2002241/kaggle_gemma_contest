import os
from tqdm import tqdm
from scrapy import Selector
from markdownify import markdownify as md


def assert_class(sel: Selector, class_name: str):
    _classes = sel.attrib["class"].split()
    assert class_name in _classes, f"Expect class {class_name}, got {_classes}"


def assert_tag(sel: Selector, tag: str):
    assert sel.root.tag == tag, f"Expect tag {tag}, got {sel.root.tag}"


def assert_tags(sel: Selector, tags: list[str]):
    for tag in tags:
        if sel.root.tag == tag:
            return
    assert sel.root.tag == tag, f"Expect tag in  {tags}, got {sel.root.tag}"


def get_text(sel: Selector, use_md=False):
    if use_md:
        text = md(str(sel), bullets="-+*", heading_style="atx", strip=["a", "script"]).strip()
    else:
        text = sel.xpath("text()").get().strip()
    return text.replace("\\*", "*").replace("\\-", "-").replace("\\.", ".")


def save(fp, paragraphs: list[dict]):
    with open(fp, "w", encoding="utf8") as f:
        for p in paragraphs:
            if p["data"].strip() != "":
                # print(f'[{p["type"]}]', file=f)
                if p["type"] in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    if p["data"].startswith("#") is False:
                        print("#" * int(p["type"][-1]), end=" ", file=f)
                print(p["data"], file=f)
                print(file=f)


def parse_html_type_1(post_xpath, filepath: str):
    with open(filepath, encoding="utf8") as f:
        html_content = f.read()

    root = Selector(text=html_content)
    paragraphs = []

    # TODO. Extract title
    title = root.xpath("/html/body/app-root/app-post-container/div[1]/div[1]/h1")
    assert len(title) == 1
    title = title[0]
    assert_class(title, "title")
    paragraphs.append({"type": "title", "data": get_text(title)})

    # TODO. Extract paragraphs    
    post = root.xpath(post_xpath)
    assert len(post) == 1, f"Not found post data at {filepath}"
    post = post[0]
    assert post.attrib["class"] == "editor"

    for ce_block in post.xpath("*"):
        assert_class(ce_block, "ce-block")
        assert_tag(ce_block, "div")

        for ce_block__content in ce_block.xpath("*"):
            assert_class(ce_block__content, "ce-block__content")
            assert_tag(ce_block__content, "div")

            for block in ce_block__content.xpath("*"):
                _classes = block.attrib["class"].split()

                if "ce-paragraph" in _classes:
                    assert_tag(block, "div")
                    paragraphs.append({
                        "type": "paragraph",
                        "data": get_text(block, use_md=True)
                    })

                elif "ce-delimiter" in _classes:
                    continue

                else:
                    sub_block = block.xpath("*")[0]
                    _classes = sub_block.attrib["class"].split()

                    if "image-tool__image" in _classes:
                        assert_tag(sub_block, "div")
                        continue

                    elif "ce-header" in _classes:
                        assert_tags(sub_block, ["h1", "h2", "h3", "h4", "h5", "h6"])
                        paragraphs.append({
                            "type": sub_block.root.tag,
                            "data": get_text(sub_block)
                        })

                    elif "cdx-quote__text" in _classes:
                        assert_tag(block, "blockquote")
                        paragraphs.append({
                            "type": "blockquote",
                            "data": get_text(sub_block)
                        })

                    elif "cdx-pull-quote__text" in _classes:
                        assert_tag(block, "blockquote")
                        paragraphs.append({
                            "type": "blockquote",
                            "data": get_text(sub_block)
                        })

                    elif "cdx-pull-quote__caption" in _classes:
                        assert_tag(block, "blockquote")
                        paragraphs.append({
                            "type": "blockquote",
                            "data": get_text(sub_block)
                        })

                    elif "link-tool" in _classes:
                        assert_tag(sub_block, "div")

                    else:
                        raise NotImplementedError(sub_block)
    
    return paragraphs


def parse_html_type_2(post_xpath, filepath: str):
    with open(filepath, encoding="utf8") as f:
        html_content = f.read()

    root = Selector(text=html_content)
    paragraphs = []

    # TODO. Extract title
    title = root.xpath("/html/body/app-root/app-post-container/div[1]/div[1]/h1")
    assert len(title) == 1, "Not found title"
    title = title[0]
    assert_class(title, "title")
    paragraphs.append({"type": "title", "data": get_text(title)})

    # TODO. Extract paragraphs    
    post = root.xpath(post_xpath)
    assert len(post) == 1, f"Not found post data at {filepath}"
    post = post[0]
    assert post.attrib["class"] == "p-content"

    for block in post.xpath("*"):
        if block.root.tag in ["div", "p"]:
            paragraphs.append({
                "type": "paragraph",
                "data": get_text(block, use_md=True)
            })

        elif block.root.tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            paragraphs.append({
                "type": block.root.tag,
                "data": get_text(block, use_md=True)
            })

        elif block.root.tag == "blockquote":
            paragraphs.append({
                "type": "blockquote",
                "data": get_text(block, use_md=True)
            })

        elif block.root.tag in ["hr", "pre", "figure"]:
            continue

        elif block.root.tag in ["ul", "ol"]:
            paragraphs.append({
                "type": "list",
                "data": get_text(block, use_md=True)
            })

        else:
            raise NotImplementedError(block.root.tag, str(block))
    
    return paragraphs
    

data_dir = "data/history/bai-dang"
save_dir = "data/parsed"

pbar = tqdm(sorted(os.listdir(data_dir)))
for file in pbar:
    pbar.set_description(file)
    filepath = os.path.join(data_dir, file)
    fn, _ = os.path.splitext(os.path.basename(filepath))
    save_path = os.path.join(save_dir, fn + ".txt")

    try:
        paragraphs = parse_html_type_1(
            "/html/body/app-root/app-post-container/div[1]/div[2]/div[1]/div[1]/new-post/div",
            filepath
        )
    except AssertionError as e:
        try:
            paragraphs = parse_html_type_1(
                "/html/body/app-root/app-post-container/div[1]/div[3]/div[1]/div[1]/new-post/div",
                filepath
            )
        except AssertionError as e:
            try:
                paragraphs = parse_html_type_2(
                    "/html/body/app-root/app-post-container/div[1]/div[3]/div[1]/div[1]/old-post/div",
                    filepath
                )
            except AssertionError as e:
                try:
                    paragraphs = parse_html_type_2(
                        "/html/body/app-root/app-post-container/div[1]/div[2]/div[1]/div[1]/old-post/div",
                        filepath
                    )
                except AssertionError as e:
                    print(e.args[0])
                    break
    
    
    save(save_path, paragraphs)
