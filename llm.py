from data import MetaData, Attribute
import json
import numpy as np
from openai import OpenAI
from typing import List, Tuple
import sys

class LLMSampler:
    def __init__(self, client: OpenAI, model: str, meta_data: MetaData):
        self.client = client
        self.model = model
        self.meta_data = meta_data
        self.features_schema = self._create_features_schema()

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

    def _features_dict_to_list(self, features: dict) -> List:
        return [features.get(f.name, 0.0) for f in self.meta_data.features]

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
                           f"The dataset has the following features:\n{nl.join([f.name + ': ' + f.description for f in self.meta_data.features])}.\n"
                           f"The dataset has the following target:\n{self.meta_data.target.name}: {self.meta_data.target.description}. "
                           f"It can take these values: {', '.join(self.meta_data.target.values)}.\n"
            },
            {
                "role": "user",
                "content": f"Give {n} rows of example data from a variety of classes in JSON format."
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1.0,
            extra_body={
                "guided_json": schema
            }
        )

        return [self._features_dict_to_list(x) for x in json.loads(response.choices[0].message.content)]

    def sample_features(self, n: int, batch_size=1) -> np.ndarray:
        X = []

        while len(X) < n:
            batch = self._sample_features_batch(batch_size)
            X.extend(batch)

        return np.array(X[:n])

    def _sample_target_single(self, x: np.ndarray, num_trials: int = 1) -> List[int]:
        schema = {
            "type": "string",
            "enum": self.meta_data.target.values
        }

        nl = '\n'
        messages = [
            {
                "role": "system",
                "content": f"You are an expert in the field of {self.meta_data.field}.\n"
                           f"Your top priority is to provide statisticians with the domain knowedge required to analyse their data. {self.meta_data.description}\n"
                           f"The dataset has the following features:\n{nl.join([f.name + ': ' + f.description for f in self.meta_data.features])}.\n"
                           f"The dataset has the following target:\n{self.meta_data.target.name}: {self.meta_data.target.description}. "
                           f"It can take these values: {', '.join(self.meta_data.target.values)}.\n"
            },
            {
                "role": "user",
                "content": f"Give the target value for the row in the dataset:\n"
                           f"{' '.join(['The ' + self.meta_data.features[i].name + ' is ' + str(x[i]) + '.' for i in range(len(x))])}"
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            n=num_trials,
            temperature=1.0,
            extra_body={
                "guided_json": schema
            }
        )

        return [self.meta_data.target.values.index(json.loads(s.message.content)) for s in response.choices]

    def sample_targets(self, X: np.ndarray, num_trials: int = 1, target_smooth: float = 0.5) -> np.ndarray:
        y = np.ones((X.shape[0], len(self.meta_data.target.values))) * target_smooth

        for i in range(X.shape[0]):
            for j in self._sample_target_single(X[i], num_trials):
                y[i, j] += 1

        return y / y.sum(axis=1, keepdims=True)

    def sample(self, n: int, batch_size: int = 10, num_trials: int = 10, target_smooth: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        X = self.sample_features(n, batch_size)
        y = self.sample_targets(X, num_trials, target_smooth)
        return X, y

def test_iris():
    meta_data = MetaData(
        name="iris",
        description="The dataset contains measurements obtained from Iris flowers, and the goal is to use these measurements to determine the species of flower.",
        field="botany",
        features=[
            Attribute("sepal_length", "The length of the sepal in cm", "float", None),
            Attribute("sepal_width", "The width of the sepal in cm", "float", None),
            Attribute("petal_length", "The length of the petal in cm", "float", None),
            Attribute("petal_width", "The width of the petal in cm", "float", None),
        ],
        target=Attribute("species", "The species of the flower", "str", ["setosa", "versicolor", "virginica"])
    )

    client = OpenAI(api_key="none", base_url=sys.argv[1])
    sampler = LLMSampler(client, sys.argv[2], meta_data)
    X, y = sampler.sample(10)
    print(X)
    print(y)

if __name__ == "__main__":
    test_iris()
