import json
import tiktoken  # 토큰 계산을 위해 사용
import numpy as np
from collections import defaultdict

data_path = "data/test.jsonl"

# 데이터셋 로드
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# 초기 데이터셋 통계
print("예제 수:", len(dataset))
print("첫 번째 예제:")
for message in dataset[0]["messages"]:
    print(message)

format_errors = defaultdict(int)

for ex in dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue
        
    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue
        
    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1
        
        if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
            format_errors["message_unrecognized_key"] += 1
        
        if message.get("role", None) not in ("system", "user", "assistant", "function"):
            format_errors["unrecognized_role"] += 1
            
        content = message.get("content", None)
        function_call = message.get("function_call", None)
        
        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1
    
    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("발견된 오류:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print('\033[95m' + "오류 없음" + '\033[37m')

encoding = tiktoken.get_encoding("cl100k_base")

# 정확하지는 않음!
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb에서 단순화함
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if isinstance(value, str):  # 문자열인지 확인
                num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            if isinstance(message["content"], str):  # 문자열인지 확인
                num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### {name} 분포:")
    print(f"최소 / 최대: {min(values)}, {max(values)}")
    print(f"평균 / 중앙값: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

# 경고 및 토큰 수 계산
n_missing_system = 0
n_missing_user = 0
n_messages = []
convo_lens = []
assistant_message_lens = []

for ex in dataset:
    messages = ex["messages"]
    if not any(message["role"] == "system" for message in messages):
        n_missing_system += 1
    if not any(message["role"] == "user" for message in messages):
        n_missing_user += 1
    n_messages.append(len(messages))
    convo_lens.append(num_tokens_from_messages(messages))
    assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
    
print("시스템 메시지가 없는 예제 수:", n_missing_system)
print("사용자 메시지가 없는 예제 수:", n_missing_user)
print_distribution(n_messages, "예제당 메시지 수")
print_distribution(convo_lens, "예제당 총 토큰 수")
print_distribution(assistant_message_lens, "예제당 어시스턴트 토큰 수")

n_too_long = sum(l > 4096 for l in convo_lens)
print(f"\n{n_too_long}개의 예제가 4096 토큰 한도를 초과할 수 있으며, 이는 미세 조정 중 잘릴 것입니다")

# 가격 및 기본 n_epochs 추정
MAX_TOKENS_PER_EXAMPLE = 4096

TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
print(f"데이터셋에는 훈련 중 청구될 ~{n_billing_tokens_in_dataset} 토큰이 있습니다")
print(f"기본적으로 이 데이터셋에서 {n_epochs} 에포크 동안 훈련하게 됩니다")
print(f"기본적으로 ~{n_epochs * n_billing_tokens_in_dataset} 토큰에 대해 청구됩니다")
