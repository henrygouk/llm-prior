import arff
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple

@dataclass
class Attribute:
    name: str
    description: str
    dtype: str
    values: List[str] | None

@dataclass
class MetaData:
    name: str
    description: str
    field: str
    features: List[Attribute]
    target: Attribute

def load_arff(data_path: str) -> Tuple[MetaData, np.ndarray, np.ndarray]:
    records = arff.load(open(data_path, 'r'))

    data = np.array(records["data"])
    X = data[:,:-1]
    y = data[:,-1]

    name_parts = records["relation"].split('#')
    field = name_parts[0].strip()
    rel_name = name_parts[1].strip()

    features = []

    for i, (schema, attrib_type) in enumerate(records["attributes"][:-1]):
        parts = schema.split(':')
        name = parts[0].strip()
        description = parts[1].strip()

        if isinstance(attrib_type, list):
            X[:, i] = [attrib_type.index(v) for v in X[:, i]]
            features.append(Attribute(name=name, description=description, dtype="str", values=attrib_type))
        else:
            features.append(Attribute(name=name, description=description, dtype="float", values=None))

    target_parts = records["attributes"][-1][0].split(':')
    target_name = target_parts[0].strip()
    target_description = target_parts[1].strip()

    target = Attribute(name=target_name, description=target_description, dtype="str", values=np.unique(y).tolist())

    meta_data = MetaData(name=rel_name, description=records["description"], field=field, features=features, target=target)

    # Convert X to float
    X = X.astype(float)

    # Convert y from string class names to integers
    y = np.array([target.values.index(y_i) for y_i in y])

    return meta_data, X, y
