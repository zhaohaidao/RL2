## Data Preparation

To run the scripts in this folder, you need to firstly download the corresponding dataset under the [RL2](https://huggingface.co/collections/ChenmieNLP/rl2-6830d8471c8d6b031d4cbc87) collection and manully convert it to JSON file.
The library accepts data with following format

* Supervised Fine-Tuning
```
[
    {
        "messages": [
            {"role": "user", "content": "What is the capital of China?"},
            {"role": "assistant", "content": "Beijing."},
            {"role": "user", "content": "What is the capital of United States?"},
            {"role": "assistant", "content": "Washington D.C."}
        ]
    }
]
```
* Direct Preference Optimization
```
[
    {
        "messages": [
            {"role": "assistant", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of China?"}
        ],
        "chosen": "Beijing.",
        "rejected": "Washington D.C."
    }
]
```
* Reinforcement Learning
```
[
    {
        "messages": [
            {"role": "assistant", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of China?"}
        ],
        "answer": "Beijing"
    }
]
```

