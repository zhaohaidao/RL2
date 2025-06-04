from math_verify import parse, verify

def reward_fn(messages, answer):
    return verify(parse(answer), parse(messages[-1]["content"]))