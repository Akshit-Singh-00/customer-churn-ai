from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

POLICY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "retention_policies"
    / "retention_policies.txt"
)


def load_policies() -> list[str]:

    text = POLICY_PATH.read_text(
        encoding="utf-8"
    )

    sections = [
        section.strip()
        for section in text.split("\n\n\n")
        if section.strip()
    ]

    return sections


POLICIES = load_policies()


vectorizer = TfidfVectorizer(
    stop_words="english"
)

policy_matrix = vectorizer.fit_transform(
    POLICIES
)


def retrieve_policies(
    query: str,
    top_k: int = 3
) -> list[str]:

    query_vector = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
        policy_matrix
    )[0]

    ranked_indices = scores.argsort()[
        ::-1
    ]

    top_indices = ranked_indices[
        :top_k
    ]

    results = []

    for index in top_indices:

        if scores[index] > 0:

            results.append(
                POLICIES[index]
            )

    return results



if __name__ == "__main__":

    query = (
        "month-to-month customer "
        "without technical support "
        "with high monthly charges"
    )

    policies = retrieve_policies(
        query,
        top_k=3
    )

    for i, policy in enumerate(
        policies,
        start=1
    ):

        print(
            f"\n--- POLICY {i} ---"
        )

        print(policy)