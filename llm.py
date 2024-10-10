from math import e
from data import MetaData, Attribute
import json
import numpy as np
from openai import OpenAI
from typing import List, Tuple
from pydantic import create_model, Field
import enum
import sys
#from vllm import LLM, SamplingParams, GuidedDecodingParams

def llm_factory(model: str, base_url: None | str = None):
    if base_url is None:
        #return LocalLLM(model)
        raise NotImplementedError("Local LLM is not supported")
    else:
        return RemoteLLM(model, base_url)

class LocalLLM:
    def __init__(self, model: str):
        self.model = model
        self.llm = LLM(model=model)

    def get_completion(self, messages: List[dict], schema: dict, max_tokens: int | None = None):
        sampling_params = SamplingParams(temperature=1.0, max_tokens=max_tokens, guided_decoding=GuidedDecodingParams(schema))
        return self.llm.chat(messages, sampling_params)[0].outputs[0].text

class RemoteLLM:
    def __init__(self, model: str, base_url: str):
        self.client = OpenAI(api_key="none", base_url=base_url)
        self.model = model

    def get_completion(self, messages: List[dict], schema: dict, max_tokens: int | None = None):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1.0,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Name",
                    "schema": schema,
                    "description": "Description",
                    "strict": True
                }
            }
        ).choices[0].message.content

class DirectLLMSampler:
    def __init__(self, meta_data: MetaData, model: str, base_url: str | None):
        self.llm = llm_factory(model, base_url)
        self.meta_data = meta_data
        self.features_schema = self._create_features_schema()
        self.target_schema = self._create_target_schema()

    def _create_features_schema(self) -> dict:
        properties = {}

        for f in self.meta_data.features:
            if f.dtype == "float":
                properties[f.name] = {"type": "number"}
            elif f.dtype == "str":
                properties[f.name] = {"type": "string", "enum": f.values}
            else:
                raise ValueError(f"Invalid data type: {f.dtype}. Must be one of ['float', 'str']")

        return {
            "type": "object",
            "properties": properties
        }

    def _create_target_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string"
                },
                "target": {
                    "type": "string",
                    "enum": self.meta_data.target.values
                }
            }
        }

    def _features_dict_to_list(self, features: dict) -> List:
        fs = [features.get(f.name, 0.0) for f in self.meta_data.features]

        # Convert string features to integers
        for i, f in enumerate(self.meta_data.features):
            if f.dtype == "str":
                fs[i] = f.values.index(fs[i])

        return fs

    def _feature_value_to_str(self, x: np.ndarray, i: int) -> str:
        f = self.meta_data.features[i]
        if f.dtype == "float":
            return str(x[i])
        elif f.dtype == "str":
            return f.values[int(x[i])]
        else:
            raise ValueError(f"Invalid data type: {f.dtype}. Must be one of ['float', 'str']")

    def _sample_features_batch(self, n: int) -> List:
        schema = {
            "type": "array",
            "items": self.features_schema
        }

        nl = '\n'
        messages = [
            {
                "role": "system",
                "content": f"You are an expert in the field of {self.meta_data.field}.\n"
                           f"Your top priority is to provide statisticians with the domain knowedge required to analyse their data. {self.meta_data.description}\n"
                           f"The dataset has the following features:\n{nl.join([self._feature_to_str(f) for f in self.meta_data.features])}\n"
                           f"The dataset has the following target:\n{self.meta_data.target.name}: {self.meta_data.target.description} "
                           f"It can take these values: {', '.join(self.meta_data.target.values)}.\n"
            },
            {
                "role": "user",
                "content": f"Give {n} rows of example data from a variety of classes in JSON format."
            }
        ]

        response = self.llm.get_completion(messages, schema)

        try:
            result = [] 

            for x in json.loads(response):
                try:
                    result.append(self._features_dict_to_list(x))
                except Exception as e:
                    print(f"Error parsing response: {response}", file=sys.stderr)
                    print(f"Error: {e}", file=sys.stderr)

            return result

        except json.JSONDecodeError as e:
            print(f"Error parsing response: {response}", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            return []

    def sample_features(self, n: int, batch_size=1) -> np.ndarray:
        X = []

        while len(X) < n:
            batch = self._sample_features_batch(min(batch_size, n - len(X)))
            X.extend(batch)
            # Print progress out to stderr
            print(f"Sampled {len(X)}/{n} examples", file=sys.stderr)

        return np.array(X[:n])

    def _feature_to_str(self, f: Attribute) -> str:
        if f.dtype == "float":
            return f"{f.name}: {f.description}"
        elif f.dtype == "str":
            return f"{f.name}: {f.description} It can take these values: {', '.join(f.values)}."
        else:
            raise ValueError(f"Invalid data type: {f.dtype}. Must be one of ['float', 'str']")

    def _sample_target_single(self, x: np.ndarray, num_trials: int = 1) -> List[int]:
        nl = '\n'
        messages = [
            {
                "role": "system",
                "content": f"You are an expert in the field of {self.meta_data.field}.\n"
                           f"Your top priority is to provide statisticians with the domain knowedge required to analyse their data. {self.meta_data.description}\n"
                           f"The dataset has the following features:\n{nl.join([self._feature_to_str(f) for f in self.meta_data.features])}\n"
                           f"The dataset has the following target:\n{self.meta_data.target.name}: {self.meta_data.target.description} "
                           f"It can take these values: {', '.join(self.meta_data.target.values)}.\n"
            },
            {
                "role": "user",
                "content": f"Give the target value for the following row in the dataset, but first explain your reasoning:\n"
                           f"{' '.join(['The ' + self.meta_data.features[i].name + ' is ' + self._feature_value_to_str(x, i) + '.' for i in range(len(x))])}\n"
                           f"Give your response in JSON format, where the first field contains a brief summary of your reasoning, and the second your prediction"
                           f" of the target"
            }
        ]

        ret = []

        #for s in [self.llm.get_completion(messages, self.target_schema, max_tokens=512) for _ in range(num_trials)]:
        while len(ret) < num_trials:
            try:
                s = self.llm.get_completion(messages, self.target_schema, max_tokens=512)
            except Exception as e:
                print(f"Error getting completion: {e}", file=sys.stderr)
                continue

            try:
                json_res = json.loads(s)
            except json.JSONDecodeError as e:
                print(f"Error parsing response: {s}", file=sys.stderr)
                print(f"Error: {e}", file=sys.stderr)
                continue

            try:
                t = json_res["target"]
                if t in self.meta_data.target.values:
                    ret.append(self.meta_data.target.values.index(t))
            except KeyError as e:
                print(f"Error parsing response: {s}", file=sys.stderr)
                print(f"Error: {e}", file=sys.stderr)

        return ret

    def sample_targets(self, X: np.ndarray, num_trials: int = 1, target_smooth: float = 0.5) -> np.ndarray:
        y = np.ones((X.shape[0], len(self.meta_data.target.values))) * target_smooth

        for i in range(X.shape[0]):
            for j in self._sample_target_single(X[i], num_trials):
                y[i, j] += 1

            print(f"Sampled targets for example {i+1}/{X.shape[0]}", file=sys.stderr)

        return y / y.sum(axis=1, keepdims=True)

    def sample(self, n: int, batch_size: int = 5, num_trials: int = 5, target_smooth: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        X = self.sample_features(n, batch_size)
        y = self.sample_targets(X, num_trials, target_smooth)
        return X, y

def test_iris():
    meta_data = MetaData(
        name="iris",
        description="The dataset contains measurements obtained from Iris flowers, and the goal is to use these measurements to determine the species of flower.",
        field="botany",
        features=[
            Attribute("sepal_length", "The length of the sepal in cm.", "float", None),
            Attribute("sepal_width", "The width of the sepal in cm.", "float", None),
            Attribute("petal_length", "The length of the petal in cm.", "float", None),
            Attribute("petal_width", "The width of the petal in cm.", "float", None),
        ],
        target=Attribute("species", "The species of the flower", "str", ["setosa", "versicolor", "virginica"])
    )

    client = OpenAI(api_key="none", base_url=sys.argv[1])
    sampler = DirectLLMSampler(client, sys.argv[2], meta_data)
    X, y = sampler.sample(10)
    print(X)
    print(y)

if __name__ == "__main__":
    test_iris()
