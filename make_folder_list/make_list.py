# ファイルの読み込み
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


# 特殊文字の変換
def repalce_special_characters(text):
    text = text.replace(" ", "_")
    text = text.replace("'", "").replace(".", "")
    text = text.replace("/", "_")
    return text


english_items = read_file('items_eng.txt')

# 特殊文字の置換を行う
modified_english_items = [
    repalce_special_characters(item) for item in english_items]

print(modified_english_items)
