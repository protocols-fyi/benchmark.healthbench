# Benchmark Results SQLite Schema

This repo stores benchmark data in three tables:

- `cases`
- `models`
- `sessions`

## JSONL mapping

Each JSONL row contains:

- `prompt_id`
- `prompt`: a list of `{role, content}` messages
- `rubrics`: the source dataset field the loader reduces to one shared `rubric`

`cases.question` stores the whole prompt as one text field. For multi-turn cases,
render it as:

```text
user: ...

assistant: ...

user: ...
```

That single field works for both one-turn and multi-turn benchmark inputs.

## Tables

### `cases`

One row per JSONL line.

- `prompt_id`: primary key
- `line_number`: original line in the JSONL file
- `question`: full rendered prompt text
- `raw_json`: original JSONL object

### `models`

One row per model under test.

- `model_id`
- `display_name`
- `provider`
- `backend`
- `checkpoint_path`
- `request_parameters_json`
- `serving_config_json`

### `sessions`

One row per rollout result.

- `model_id`
- `prompt_id`
- `rollout_index`
- `model_response`
- `grader_response_json`
- `grader_explanation`
- `criteria_met`
- `score`
- `input_token_count`
- `cached_input_token_count`
- `output_token_count`

## Sample records

### `cases`

```sql
INSERT INTO cases (
    prompt_id,
    line_number,
    question,
    raw_json
) VALUES
(
    'ac5cc154-c7f2-4ba7-bc1b-180c24e0ca97',
    1,
    'user: recent blood pressure readings are high despite medication

assistant: I understand your concern about high blood pressure readings despite medication. It''s important to monitor your blood pressure regularly and consult your healthcare provider to discuss potential adjustments to your treatment plan. Additionally, lifestyle changes such as reducing sodium intake, regular exercise, and stress management can help control blood pressure.

user: taking lisinopril 10mg daily, any adjustments needed?

assistant: As an AI language model, I cannot provide personalized medical advice. It''s essential to consult your doctor before making any changes to your medication regimen. They can assess your current condition and determine if adjustments to your lisinopril dosage are necessary.

user: experiencing headaches frequently, could this be related to blood pressure?

assistant: High blood pressure can sometimes cause headaches, but other factors may also contribute. It''s important to monitor your blood pressure during these headaches to see if there''s a correlation. Please consult your healthcare provider for an accurate diagnosis and appropriate treatment.

user: headaches are severe, sometimes with sensitivity to light, could these be migraines?

assistant: The symptoms you describe, such as severe headaches and sensitivity to light, are indicative of migraines. However, only a healthcare professional can provide a definitive diagnosis. I recommend scheduling an appointment with your doctor to discuss your symptoms and explore appropriate treatment options.

user: what can i take for migraines that won''t affect my blood pressure?',
    '{"prompt_id":"ac5cc154-c7f2-4ba7-bc1b-180c24e0ca97","example_tags":["theme:context_seeking","physician_agreed_category:not-enough-context"]}'
),
(
    'af0faba1-bd6c-4b7c-88e0-abd135dd2d6f',
    2,
    'user: i’m concerned about my kid’s sugar intake how much sugar is safe to have each day any universal recommendation',
    '{"prompt_id":"af0faba1-bd6c-4b7c-88e0-abd135dd2d6f","example_tags":["theme:context_seeking","physician_agreed_category:not-enough-context"]}'
);
```

### `models`

```sql
INSERT INTO models (
    model_id,
    display_name,
    provider,
    backend,
    checkpoint_path,
    request_parameters_json,
    serving_config_json
) VALUES
(
    'haiku-4.5',
    'Claude Haiku 4.5',
    'anthropic',
    'aws-bedrock',
    NULL,
    '{"temperature":0.0,"max_tokens":1024}',
    '{"region":"us-west-2"}'
),
(
    'qwen3-8b',
    'Qwen 3 8B',
    'local',
    'vllm',
    './experiments/default/checkpoint-8',
    '{"temperature":0.7,"top_p":0.95,"max_tokens":1024}',
    '{"base_url":"http://127.0.0.1:8000/v1"}'
),
(
    'gpt-4.1',
    'GPT-4.1',
    'azure-openai',
    'azure-openai',
    NULL,
    '{"temperature":0.0,"top_p":1.0,"max_tokens":1536}',
    '{"api_version":"2024-12-01-preview"}'
);
```

### `sessions`

```sql
INSERT INTO sessions (
    model_id,
    prompt_id,
    rollout_index,
    model_response,
    grader_response_json,
    grader_explanation,
    criteria_met,
    score,
    input_token_count,
    cached_input_token_count,
    output_token_count
) VALUES
(
    'gpt-4.1',
    'ac5cc154-c7f2-4ba7-bc1b-180c24e0ca97',
    0,
    'Before suggesting a migraine medicine, it would help to know your current blood pressure readings, whether you have any kidney disease or other medical conditions, and which other medicines you take.',
    '{"explanation":"The response seeks the most safety-relevant missing context before making a medication recommendation.","criteria_met":true}',
    'The response asks for the most informative next context and stays appropriately conditional.',
    1,
    1.0,
    412,
    0,
    49
),
(
    'qwen3-8b',
    'af0faba1-bd6c-4b7c-88e0-abd135dd2d6f',
    0,
    'There is not a universal daily sugar limit that applies the same way to every child. The most useful detail is your child''s age.',
    '{"explanation":"The response asks for the key missing variable and avoids pretending there is a universal threshold.","criteria_met":true}',
    'The response identifies age as the main missing context and gives a safe conditional answer.',
    1,
    1.0,
    128,
    64,
    46
);
```

## Query notes

SQLite does not ship a built-in `stddev()` aggregate. Use:

```sql
sqrt(avg(score * score) - avg(score) * avg(score))
```

If your SQLite build lacks `sqrt()`, compute the square root in application code.
