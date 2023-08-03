# currently we answer immediately
sensor_query_base_prompt = """Does this text contain {}?
```
{}
```"""


def get_sensor_query(sensor: str, final_text: str):
    return [
        {"role": "user", "content": sensor_query_base_prompt.format(sensor, final_text)},
    ]


sensor_query_no_function_prompt = """Does this text contain {}?
```
{}
```

Answer immediately with just Yes/No."""


def get_sensor_query_no_function(sensor: str, final_text: str):
    return [
        {"role": "user", "content": sensor_query_no_function_prompt.format(sensor, final_text)},
    ]
