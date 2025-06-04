import re
import string
import requests

def interact(messages):

    env_messages = []
    for query in re.findall(
        f"<search>(.*?)</search>", messages[-1]["content"]
    ):
        result = requests.post(
            "http://localhost:8000/search", json={
                "query": query
            }
        )
        env_messages.append(
            {"role": "tool", "content": result}
        )
        
    return env_messages

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def reward_fn(messages, answer):

    preds = re.findall(
        r"<answer>(.*?)</answer>", messages[-1]["content"]
    )
    if len(preds) == 0:
        return False
    pred = normalize_answer(preds[-1])

    if isinstance(answer, str):
        answer = [answer]
    answer = [normalize_answer(a) for a in answer]
  
    return pred in answer
