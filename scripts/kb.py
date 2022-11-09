import hashlib
import logging
import os.path
import pickle
from pathlib import Path
from typing import Iterable, Iterator, Tuple, Union, Optional, Dict, Any, List

import annoy
import numpy
import spacy
import srsly
import tqdm
from spacy import Vocab, Language
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span
from spacy.util import SimpleFrozenList

from extraction.utils import establish_db_connection, load_entities


class WikiKB(KnowledgeBase):
    """Knowledge base handling storage and access to Wikidata/Wikipedia data."""

    def __init__(
        self,
        vocab: Vocab,
        db_path: Path,
        annoy_path: Path,
        language: str,
        n_trees: int = 50,
    ):
        """Initializes from existing SQLite database generated by `wikid`.
        Loads Annoy index file (as mmap) into memory, if file exists at specified path.
        vocab (Vocab): Pipeline vocabulary.
        db_path (Path): Path to SQLite database.
        annoy_path (Path): Path to Annoy index file.
        language (str): Language.
        n_trees (int): Number of trees in Annoy index. Precision in NN queries correspond with number of trees. Ignored
            if Annoy index file already exists.
        """
        super().__init__(vocab, vocab.vectors_length)

        self._paths = {"db": db_path, "annoy": db_path.parent / "wiki.annoy"}
        self._language = language
        self._annoy: Optional[annoy.AnnoyIndex] = None
        self._n_trees = n_trees
        self._db_conn = establish_db_connection(language)
        self._embedding_dim = self.entity_vector_length
        self._hashes: Dict[str, Optional[str]] = {}

        if os.path.exists(self._paths["annoy"]):
            self._init_annoy_from_file()

        # todo set up everything needed to integrate WikiKB into benchmark
        #   - get_candidates:
        #       - NN search with annoy
        #       - fuzzy match in label + aliases
        #       - BM25 search in descriptions

    def build_embeddings_index(self, nlp: Language) -> None:
        """Constructs index for embeddings with Annoy and stores them in an index file.
        nlp (Language): Pipeline with tok2vec for inferring embeddings.
        """

        logger = logging.getLogger(__name__)

        # Initialize ANN index.
        self._annoy = annoy.AnnoyIndex(self._embedding_dim, "angular")
        self._annoy.on_disk_build(str(self._paths["annoy"]))
        batch_size = 100000

        row_count = (
            self._db_conn.cursor()
            .execute("SELECT count(*) FROM entities")
            .fetchone()["count(*)"]
        )
        for row_id in tqdm.tqdm(
            # We select by ROWID, which starts at 1.
            range(1, row_count + 1, batch_size),
            desc="Inferring entity embeddings",
            position=0,
        ):
            ids = tuple(
                (row["id"], row["ROWID"])
                for row in self._db_conn.cursor()
                .execute(
                    f"""
                        SELECT
                            id,
                            ROWID
                        FROM
                            entities
                        WHERE
                            ROWID BETWEEN {row_id} AND {row_id + batch_size - 1}
                        ORDER BY
                            ROWID
                        """
                )
                .fetchall()
            )
            qids = tuple(_id[0] for _id in ids)
            entities = load_entities(language=self._language, qids=qids)

            ent_descs = [
                " ".join({entities[qid].name, *entities[qid].aliases})
                + " "
                + (
                    entities[qid].description
                    if entities[qid].description
                    else (
                        entities[qid].article_text[:500]
                        if entities[qid].article_text
                        else ""
                    )
                )
                for qid in qids
            ]

            for row_id_offset, qid, desc_vector in zip(
                range(len(qids)),
                qids,
                [
                    ent_desc_doc.vector
                    for ent_desc_doc in nlp.pipe(texts=ent_descs, n_process=-1)
                ],
            ):
                self._annoy.add_item(
                    # Annoy expects index to start with 0, so we index each entities vector by its entities.ROWID value
                    # in the database shifted by -1.
                    row_id + row_id_offset - 1,
                    desc_vector
                    if isinstance(desc_vector, numpy.ndarray)
                    else desc_vector.get(),
                )

        logger.info("Building ANN index.")
        self._annoy.build(self._n_trees, -1)

    def get_candidates_all(
        self, mentions: Iterator[Iterable[Span]]
    ) -> Iterator[Iterable[Iterable[Candidate]]]:
        """
        Return candidate entities for specified mentions. Each candidate defines the qid, the original alias,
        and the prior probability of that alias resolving to that qid.
        If no candidate is found for a given mention, an empty list is returned.
        mentions (Generator[Iterable[Span]]): Mentions per documents for which to get candidates.
        RETURNS (Generator[Iterable[Iterable[Candidate]]]): Identified candidates per document.
        """

        # todo
        #   - alt 1:
        #       1. fuzzy alias search
        #       2. FTS search
        #   - alt 2:
        #       1. embedding ANN search
        #   - for both: beam search
        # todo assemble one query with UNION ALL and additional (constant) ID column.
        # todo consider entity-alias prior count in ranking/at least extract them.
        for doc_mentions in mentions:
            pass
            # start = time.time()
            # alias_hits = [
            #     dict(row)
            #     for row in self._db_conn.execute(
            #         """
            #         SELECT
            #             *
            #         FROM (
            #             SELECT alias, 0 as distance, null as score FROM ALIASES_FOR_ENTITIES where alias = ?
            #             UNION
            #             SELECT word, distance, score FROM aliases WHERE word MATCH ?
            #         )
            #         ORDER BY
            #             score
            #         LIMIT 100
            #         """,
            #         ((mention.text, mention.text) for mention in doc_mentions),
            #     ).fetchall()
            # ]
            # duration_alias = time.time() - start
            # start = time.time()
            # fts_hits = [
            #     dict(row) for row in
            #     self._db_conn.execute(
            #         f"""
            #         SELECT
            #             bm25(entities_texts), et.*
            #         FROM
            #             entities_texts et
            #         WHERE
            #             entities_texts MATCH '{mention.text}'
            #         ORDER BY
            #             bm25(entities_texts)
            #         LIMIT 100
            #         """,
            #     ).fetchall()
            # ]
            # duration_fts = time.time() - start
            # x = 3

            # yield [self.get_candidates(span) for span in doc_mentions]

    def get_candidates(self, mention: Span) -> Iterable[Candidate]:
        """
        Not supported.
        mention (Span): Mention for which to get candidates.
        RETURNS (Iterable[Candidate]): Identified candidates.
        """
        raise NotImplementedError

    def get_vectors(self, qids: Iterable[str]) -> Iterable[Iterable[float]]:
        """
        Return vectors for qids.
        qids (str): Wiki QIDs.
        RETURNS (Iterable[Iterable[float]]): Vectors for specified entities.
        """
        return [self.get_vector(qid) for qid in qids]

    def get_vector(self, qid: str) -> Iterable[float]:
        """
        Return vector for qid.
        qid (str): Wiki QID.
        RETURNS (Iterable[float]): Vector for specified entities.
        """
        raise NotImplementedError

    @staticmethod
    def _hash_file(file_path: Path, blocksize: int = 2**20) -> str:
        """Generates MD5 file of hash iteratively (without loading entire file into memory.
        Source: https://stackoverflow.com/a/1131255.
        file_path (Path): Path of file to hash.
        blocksize (int): Size of blocks to load into memory (in bytes).
        RETURN (str): MD5 hash of file.
        """
        file_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read(blocksize)
            while buf:
                file_hash.update(buf)
                buf = f.read(blocksize)
        return file_hash.hexdigest()

    def _init_annoy_from_file(self) -> None:
        """Inits Annoy index."""
        self._annoy = annoy.AnnoyIndex(self._embedding_dim, "angular")
        self._annoy.load(str(self._paths["annoy"]))

    def _update_hash(self, key: str) -> str:
        """Updates hash.
        key (str): Key for file to hash - has to be in self._paths.
        RETURNS (str): File hash.
        """
        self._hashes[key] = self._hash_file(self._paths[key])
        return self._hashes[key]

    def to_bytes(self, **kwargs) -> bytes:
        """Serialize the current state to a binary string.
        RETURNS (bytes): Current state as binary string.
        """
        return spacy.util.to_bytes(
            {
                "meta": lambda: srsly.json_dumps(
                    data=(
                        self._language,
                        {key: str(path) for key, path in self._paths.items()},
                        self._embedding_dim,
                        self._update_hash("db"),
                        self._update_hash("annoy"),
                    )
                ).encode("utf-8"),
                "vocab": self.vocab.to_bytes,
            },
            [],
        )

    def from_bytes(
        self, bytes_data: bytes, *, exclude: Tuple[str] = tuple()
    ) -> "WikiKB":
        """Load state from a binary string.
        bytes_data (bytes): KB state.
        exclude (Tuple[str]): Properties to exclude when restoring KB.
        """

        def deserialize_meta(value: bytes) -> None:
            """De-serialize meta info.
            value (bytes): Byte string to deserialize.
            """
            meta_info = srsly.json_loads(value)
            self._language = meta_info[0]
            self._paths = {k: Path(v) for k, v in meta_info[1].items()}
            self._embedding_dim = meta_info[2]
            self._hashes["db"] = meta_info[3]
            self._hashes["annoy"] = meta_info[4]
            self._init_annoy_from_file()
            for file_id in ("annoy", "db"):
                assert self._hashes[file_id] == self._hash_file(
                    self._paths[file_id]
                ), f"File with internal ID {file_id} does not match deserialized hash."

        def deserialize_vocab(value: bytes):
            """De-serialize vocab.
            value (bytes): Byte string to deserialize.
            """
            self.vocab.from_bytes(value)

        spacy.util.from_bytes(
            bytes_data, {"meta": deserialize_meta, "vocab": deserialize_vocab}, exclude
        )

        return self

    def to_disk(
        self, path: Union[str, Path], exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """
        Write WikiKnowledgeBase content to disk.
        path (Union[str, Path]): Target file path.
        exclude (Iterable[str]): List of components to exclude.
        """
        path = spacy.util.ensure_path(path)
        if not path.exists():
            path.mkdir(parents=True)
        if not path.is_dir():
            raise ValueError(spacy.Errors.E928.format(loc=path))

        def pickle_data(value: Any, file_path: Path) -> None:
            """
            Pickles info to disk.
            value (Any): Value to pickle.
            file_path (Path): File path.
            """
            with open(file_path, "wb") as file:
                pickle.dump(value, file)

        self._update_hash("db")
        self._update_hash("annoy")

        serialize = {
            "meta": lambda p: pickle_data(
                (self._language, self._paths, self._embedding_dim, self._hashes), p
            ),
            "vocab.json": lambda p: self.vocab.strings.to_disk(p),
        }
        spacy.util.to_disk(path, serialize, exclude)

    def from_disk(
        self, path: Union[str, Path], exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """
        Load WikiKnowledgeBase content from disk.
        path (Union[str, Path]): Target file path.
        exclude (Iterable[str]): List of components to exclude.
        """
        path = spacy.util.ensure_path(path)
        if not path.exists():
            raise ValueError(spacy.Errors.E929.format(loc=path))
        if not path.is_dir():
            raise ValueError(spacy.Errors.E928.format(loc=path))

        def deserialize_meta_info(file_path: Path) -> None:
            """
            Deserializes meta info.
            file_path (Path): File path.
            RETURNS (Any): Deserializes meta info.
            """
            with open(file_path, "rb") as file:
                meta_info = pickle.load(file)
                self._language = meta_info[0]
                self._paths = meta_info[1]
                self._embedding_dim = meta_info[2]
                self._hashes = meta_info[3]
                self._init_annoy_from_file()
                for file_id in ("annoy", "db"):
                    assert self._hashes[file_id] == self._hash_file(
                        self._paths[file_id]
                    ), f"File with internal ID {file_id} does not match deserialized hash."

        deserialize = {
            "meta": lambda p: deserialize_meta_info(p),
            "vocab.json": lambda p: self.vocab.strings.from_disk(p),
        }
        spacy.util.from_disk(path, deserialize, exclude)

    @staticmethod
    def _pick_candidate_sequences(
        embeddings: numpy.ndarray, beam_width: int
    ) -> List[Tuple[List[int], float]]:
        """Pick sequences of candidates, ranked by their cohesion. Cohesion is measured as the average cosine similarity
        between the average embedding in a sequence and the individual embeddings.
        Each row contains all candidates per mention. Selects heuristically via beam search.
        Modified from https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/.
        embeddings (numpy.ndarray): 2D matrix with embedding vectors per candidate.
        beam_width (int): Beam width.
        RETURN (List[Tuple[List[int], float]]): List of sequences of candidate indices in embeddings matrix & their
            corresponding cohesion score.
        """
        # todo add shape check (3D) for embeddings
        # todo step-by-step debugging with real data to verify assumptions
        # todo ensure correct shape processing in numpy operations
        sequences: List[Tuple[List[int], float]] = [([], 0.0)]
        dim = len(embeddings[0][0])

        for row_idx, row in enumerate(embeddings):
            all_candidates: List[Tuple[List[int], float]] = []
            # Expand each candidate.
            for i in range(len(sequences)):
                sequence = sequences[i]
                # Compute sum of embeddings already in this sequence. If this is the first row and `sequences` is hence
                # empty, we assume a vector of zeroes.
                seq_prev_embeddings = [
                    embeddings[_row_idx][col_idx]
                    for _row_idx, col_idx in enumerate(sequence[0])
                ]
                seq_sum_prev_embeddings = (
                    numpy.sum(seq_prev_embeddings) if row_idx > 0 else numpy.zeros(dim)
                )

                for j in range(len(row)):
                    # Compute average sequence embedding, including potential next sequence element.
                    seq_avg_embedding = numpy.sum(seq_sum_prev_embeddings, row[j]) / (
                        row_idx + 1
                    )
                    seq_embeddings = [*seq_prev_embeddings, row[j]]

                    # Compute cohesion as cosine similarity.
                    cohesion = numpy.mean(
                        (seq_avg_embedding @ seq_embeddings)
                        / (
                            numpy.linalg.norm(seq_avg_embedding)
                            * numpy.linalg.norm(seq_embeddings)
                        )
                    )
                    candidate = [(*sequence[0], j), cohesion]
                    all_candidates.append(candidate)

            # Order all candidates by cohesion, select beam_width best sets.
            sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]  # type: ignore

        return sequences
