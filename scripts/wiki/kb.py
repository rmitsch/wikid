import logging
from pathlib import Path
from typing import Iterable, Iterator, Tuple, Union, Optional

import annoy
import numpy
import tqdm
from spacy import Vocab, Language
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span
from spacy.util import SimpleFrozenList


class WikiKB(KnowledgeBase):
    """Knowledge base handling storage and access to Wikidata/Wikipedia data."""

    def __init__(
        self,
        vocab: Vocab,
        vector_length: int,
        db_path: Path,
        language: str,
        n_trees: int = 50,
    ):
        """Initializes from existing SQLite database generated by `wikid`.
        vocab (Vocab): Pipeline vocabulary.
        vector_vector_length (int): Vector length.
        db_path (Path): Path to SQLite database.
        n_trees (int): Number of trees in Annoy index. Precision in NN queries correspond with number of trees.
        """
        super().__init__(vocab, vocab.vectors_length)
        self._paths = {"db": db_path, "vectors_ann": db_path.parent / "wiki.annoy"}
        self._language = language
        self._vector_length = vector_length
        self._vectors_ann: Optional[annoy.AnnoyIndex] = None
        self._n_trees = n_trees

        # todo set up everything needed to integrate WikiKB into benchmark
        #   - init from  database
        #   - replace old-style KB with WikiKB
        #   - implement serialization methods
        #   - start with dummy method returning nothing
        #   - NN search with annoy
        #   - exact match in label + aliases
        #   - BM25 search in descriptions
        #   - later: add fuzzy matching

    def infer_embeddings(self, nlp: Language) -> None:
        """Constructs index for vector ANN from database and stores them in an Annoy index.
        nlp (Language): Pipeline with tok2vec for inferring embeddings.
        """

        logger = logging.getLogger(__name__)
        logger.info("Loading entities")
        from . import (
            load_entities,
        )  # todo refactor so that local import isn't necessary + refactor wiki source in

        #   general (which subdirs?).
        # todo load entities batch-wise in loop (load QIDs first, order by ROWID)
        entities = load_entities(language=self._language)
        qids = list(entities.keys())

        # Construct ANN index.
        self._vectors_ann = annoy.AnnoyIndex(self._vector_length, "dot")
        self._vectors_ann.on_disk_build(str(self._paths["vectors_ann"]))
        batch_size = 100000
        i = 0

        for batch_i in tqdm.tqdm(
            range(0, len(qids), batch_size),
            desc="Inferring entity embeddings",
            position=0,
        ):
            batch_qids = qids[batch_i : batch_i + batch_size]
            texts = [
                entities[qid].name
                + " "
                + (" ".join(entities[qid].aliases) if entities[qid].aliases else "")
                + " "
                + (
                    entities[qid].description
                    if entities[qid].description
                    else (
                        entities[qid].article_text[:500]
                        if entities[qid].article_text
                        else entities[qid].name
                    )
                )
                for qid in batch_qids
            ]

            for qid, desc_vector in zip(
                batch_qids, [doc.vector for doc in nlp.pipe(texts=texts, n_process=-1)]
            ):
                self._vectors_ann.add_item(
                    i,
                    desc_vector
                    if isinstance(desc_vector, numpy.ndarray)
                    else desc_vector.get(),
                )
                i += 1

        logger.info("Building ANN index.")
        self._vectors_ann.build(self._n_trees, -1)

    def get_candidates_all(
        self, mentions: Iterator[Iterable[Span]]
    ) -> Iterator[Iterable[Iterable[Candidate]]]:
        """
        Return candidate entities for specified texts. Each candidate defines the entity, the original alias,
        and the prior probability of that alias resolving to that entity.
        If no candidate is found for a given text, an empty list is returned.
        mentions (Generator[Iterable[Span]]): Mentions per documents for which to get candidates.
        RETURNS (Generator[Iterable[Iterable[Candidate]]]): Identified candidates per document.
        """
        for doc_mentions in mentions:
            yield [self.get_candidates(span) for span in doc_mentions]

    def get_candidates(self, mention: Span) -> Iterable[Candidate]:
        """
        Return candidate entities for specified text. Each candidate defines the entity, the original alias,
        and the prior probability of that alias resolving to that entity.
        If the no candidate is found for a given text, an empty list is returned.
        mention (Span): Mention for which to get candidates.
        RETURNS (Iterable[Candidate]): Identified candidates.
        """
        raise NotImplementedError

    def get_vectors(self, entities: Iterable[str]) -> Iterable[Iterable[float]]:
        """
        Return vectors for entities.
        entity (str): Entity name/ID.
        RETURNS (Iterable[Iterable[float]]): Vectors for specified entities.
        """
        return [self.get_vector(entity) for entity in entities]

    def get_vector(self, entity: str) -> Iterable[float]:
        """
        Return vector for entity.
        entity (str): Entity name/ID.
        RETURNS (Iterable[float]): Vector for specified entity.
        """
        raise NotImplementedError

    def to_bytes(self, **kwargs) -> bytes:
        """Serialize the current state to a binary string.
        RETURNS (bytes): Current state as binary string.
        """
        raise NotImplementedError

    def from_bytes(self, bytes_data: bytes, *, exclude: Tuple[str] = tuple()):
        """Load state from a binary string.
        bytes_data (bytes): KB state.
        exclude (Tuple[str]): Properties to exclude when restoring KB.
        """
        raise NotImplementedError

    def to_disk(
        self, path: Union[str, Path], exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """
        Write KnowledgeBase content to disk.
        path (Union[str, Path]): Target file path.
        exclude (Iterable[str]): List of components to exclude.
        """
        raise NotImplementedError

    def from_disk(
        self, path: Union[str, Path], exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """
        Load KnowledgeBase content from disk.
        path (Union[str, Path]): Target file path.
        exclude (Iterable[str]): List of components to exclude.
        """
        raise NotImplementedError
