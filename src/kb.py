import hashlib
import logging
import os.path
import pickle
import sqlite3

from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator, Tuple, Union, Optional, Dict, Any, List, Sized

import annoy
import numpy
import spacy
import srsly
import tqdm
import wasabi
from spacy import Vocab, Language
from spacy.kb import KnowledgeBase, Candidate
from spacy.tokens import Span
from spacy.util import SimpleFrozenList, SimpleFrozenDict

from extraction.utils import (
    establish_db_connection,
    load_entities,
)


# Iterable of entity candidates for a single mention.
_MentionCandidates = Iterable[Candidate]
# Iterable of _MentionCandidates for a single doc.
_DocCandidates = Iterable[_MentionCandidates]


class WikiKB(KnowledgeBase):
    """Knowledge base handling storage and access to Wikidata/Wikipedia data."""

    def __init__(
        self,
        vocab: Vocab,
        entity_vector_length: int,
        db_path: Path,
        annoy_path: Path,
        language: str,
        n_trees: int = 25,
        top_k_aliases: int = 5,
        top_k_entities_alias: int = 20,
        top_k_entities_fts: int = 5,
        threshold_alias: int = 100,
        establish_db_connection_at_init: bool = True,
    ):
        """Initializes from existing SQLite database generated by `wikid`.
        Loads Annoy index file (as mmap) into memory, if file exists at specified path.
        vocab (Vocab): Pipeline vocabulary.
        entity_vector_length (int): Length of entity vectors.
        db_path (Path): Path to SQLite database.
        annoy_path (Path): Path to Annoy index file.
        language (str): Language.
        n_trees (int): Number of trees in Annoy index. Precision in NN queries correspond with number of trees. Ignored
            if Annoy index file already exists.
        top_k_aliases (int): Top k aliases matches to consider. An alias may be associated with more than one entity, so
            this parameter does _not_ necessarily correspond to the the maximum number of identified candidates. For
            that use top_k_entities_alias.
        top_k_entities_alias (int): Top k entities to consider in list of alias matches. Equals maximum number of
            candidate entities found via alias search.
        top_k_entities_fts (int): Top k of full-text search matches to consider. Equals maximum number of candidate entities
            found via full-text search.
        threshold_alias (int): Threshold for alias distance as calculated by spellfix1.
        establish_db_connection_at_init (bool): Whether to establish a DB connection on instance initialization. It
            might make sense to set this to False when the DB path isn't known at initialization time. If this is False,
            WikiKB.establish_db_connection() has to be called explicitly before any other operations are invoked.
        """
        super().__init__(vocab, entity_vector_length)

        self._paths = {"db": db_path, "annoy": annoy_path}
        self._language = language
        self._annoy: Optional[annoy.AnnoyIndex] = None
        self._n_trees = n_trees
        self._db_conn: Optional[sqlite3.Connection] = None
        self._hashes: Dict[str, Optional[str]] = {}
        self._top_k_aliases = top_k_aliases
        self._top_k_entities_alias = top_k_entities_alias
        self._top_k_entities_fts = top_k_entities_fts
        self._threshold_alias = threshold_alias

        if establish_db_connection_at_init:
            self.establish_db_connection()
        if os.path.exists(self._paths["annoy"]):
            self._init_annoy_from_file()

    def establish_db_connection(self) -> None:
        """Establishes connection to database."""
        self._db_conn = establish_db_connection(self._language, self._paths["db"])

    def _ensure_db_connection(self) -> None:
        """Ensures DB connection is set. If not, a ValueError is raised."""
        if self._db_conn is None:
            raise ValueError(
                "DB connection must be set before this operation is invoked."
            )

    def build_embeddings_index(
        self, nlp: Language, n_jobs: int = -1, batch_size: int = 2**14
    ) -> None:
        """Constructs index for embeddings with Annoy and stores them in an index file.
        nlp (Language): Pipeline with tok2vec for inferring embeddings.
        n_jobs (int): Number of jobs to use for inferring entity embeddings and building the index.
        batch_size (int): Number of entities to request at once.
        """
        self._ensure_db_connection()
        logger = logging.getLogger(__name__)

        if self._annoy:
            wasabi.msg.fail(
                title="Embeddings index already exists.",
                text=f"Delete {self._paths['annoy']} manually or with `spacy project run delete_embeddings_index` to "
                f"generate new index.",
                exits=1,
            )

        # Initialize ANN index.
        self._annoy = annoy.AnnoyIndex(self.entity_vector_length, "angular")
        self._annoy.on_disk_build(str(self._paths["annoy"]))

        row_count = (
            self._db_conn.cursor()
            .execute("SELECT count(*) FROM entities")
            .fetchone()["count(*)"]
        )

        # Build Annoy index in batches.
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
            entities = load_entities(
                language=self._language, qids=qids, db_conn=self._db_conn
            )

            # Assemble descriptions to be embedded.
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
                    for ent_desc_doc in nlp.pipe(texts=ent_descs, n_process=n_jobs)
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
        self._annoy.build(n_trees=self._n_trees, n_jobs=n_jobs)

    def get_candidates_all(
        self, mentions: Iterator[Iterable[Span]]
    ) -> Iterator[_DocCandidates]:
        """
        Retrieve candidate entities for specified mentions per document. If no candidate is found for a given mention,
        an empty list is returned.
        mentions (Iterator[Iterable[Span]]): Mentions per documents for which to get candidates.
        YIELDS (Iterator[_DocCandidates]): Identified candidates per document.
        """
        self._ensure_db_connection()

        for mentions_in_doc in mentions:
            mentions_in_doc = tuple(mentions_in_doc)
            alias_matches = self._fetch_candidates_by_alias(mentions_in_doc)
            fts_matches = self._fetch_candidates_by_fts(mentions_in_doc)
            # Candidates for each mention per document.
            candidates: List[List[Candidate]] = []

            for i, mention in enumerate(mentions_in_doc):
                candidates.append([])

                for cand_data in alias_matches.get(mention.text, []):
                    candidates[i].append(
                        Candidate(
                            kb=self,
                            entity_freq=cand_data["sum_occurence_count"],
                            prior_prob=cand_data["max_prior_prob"],
                            entity_vector=next(
                                iter(self._get_vectors([cand_data["rowid"]]))
                            ),
                            # Hashes aren't used by WikiKB.
                            entity_hash=0,
                            alias_hash=0,
                        )
                    )

                for cand_data in fts_matches.get(mention.text, []):
                    candidates[i].append(
                        Candidate(
                            kb=self,
                            entity_freq=cand_data["sum_occurence_count"],
                            prior_prob=-1,
                            entity_vector=next(
                                iter(self._get_vectors([cand_data["rowid"]]))
                            ),
                            # Hashes aren't used by WikiKB.
                            entity_hash=0,
                            alias_hash=0,
                        )
                    )

            yield candidates

    def get_candidates(self, mention: Span) -> Iterable[Candidate]:
        """
        Retrieve candidate entities for specified mention. If no candidate is found for a given mention, an empty list
        is returned.
        mention (Span): Mention for which to get candidates.
        RETURNS (Iterable[Candidate]): Identified candidates.
        """
        return next(iter(next(self.get_candidates_all([mention]))))

    def _get_vectors(self, rowids: Iterable[int]) -> Iterable[Iterable[float]]:
        """
        Return vectors for entities.
        rowids (Iterable[int]): ROWID values for entities in table `entities`.
        RETURNS (Iterable[Iterable[float]]): Vectors for specified entities.
        """
        # Annoy doesn't seem to offer batched retrieval.
        return [self._annoy.get_item_vector(rowid - 1) for rowid in rowids]

    def get_vectors(self, qids: Iterable[str]) -> Iterable[Iterable[float]]:
        """
        Return vectors for entities.
        qids (str): Wiki QIDs.
        RETURNS (Iterable[Iterable[float]]): Vectors for specified entities.
        """
        self._ensure_db_connection()
        if not isinstance(qids, Sized):
            qids = set(qids)

        # Fetch row IDs for QIDs, resolve to vectors in Annoy index.
        return self._get_vectors(
            [
                row["ROWID"]
                for row in self._db_conn.cursor()
                .execute(
                    "SELECT ROWID FROM entities WHERE id in (%s)"
                    % ",".join("?" * len(qids)),
                    tuple(qids),
                )
                .fetchall()
            ]
        )

    def get_vector(self, qid: str) -> Iterable[float]:
        """
        Return vector for qid.
        qid (str): Wiki QID.
        RETURNS (Iterable[float]): Vector for specified entities.
        """
        return next(iter(self.get_vectors([qid])))

    def _fetch_candidates_by_alias(
        self, mentions: Tuple[Span, ...]
    ) -> Dict[str, List[Dict[str, Union[str, int, float]]]]:
        """Fetches candidates for mentions by fuzzily matching aliases to the mentions.
        mentions (Tuple[Span, ..]): List of mentions for which to fetch candidates.
        RETURN List[Dict[str, Dict[str, Union[str, int, float]]]]: List of candidates per mention, sorted by distance
            to mention. Each candidate entry includes:
            - entity ID,
            - maximum prior probability over all aliases per entity,
            - sum of occurences in Wikipedia over all aliases per entity,
            - min. lexical distance over all aliases per entity,
            - row ID of entity (relevant for linking to other tables).
        """
        # Subquery to fetch alias values for single mention.
        mention_subquery = f"""
            SELECT alias, 0 as distance, null as score FROM aliases_for_entities WHERE alias = ?
            UNION
            SELECT word, distance, score FROM aliases WHERE word MATCH ? AND distance <= {self._threshold_alias}
        """

        grouped_rows: Dict[str, List[Dict[str, Union[str, int, float]]]] = {}
        for grouped_row in [
            dict(row)
            for row in self._db_conn.execute(
                """
                SELECT
                    matches.mention,
                    matches.entity_id,
                    matches.max_prior_prob,
                    matches.sum_occurence_count,
                    matches.min_distance,
                    e.ROWID
                FROM ("""
                + "\nUNION ALL\n".join(
                    [
                        f"""
                            SELECT
                                *
                            FROM (
                                SELECT
                                    matches.mention,
                                    ae.entity_id,
                                    max(ae.prior_prob) as max_prior_prob,
                                    sum(ae.count) as sum_occurence_count,
                                    min(matches.distance) as min_distance
                                FROM (
                                    SELECT
                                        ? as mention,
                                        matches.alias,
                                        matches.distance
                                    FROM
                                        ({mention_subquery}) matches
                                    ORDER BY
                                        score
                                    LIMIT {self._top_k_aliases}
                                ) matches
                                INNER JOIN aliases_for_entities ae on
                                    ae.alias = matches.alias
                                GROUP BY
                                    ae.entity_id
                                ORDER BY
                                    min_distance,
                                    sum_occurence_count DESC
                                LIMIT {self._top_k_entities_alias}
                            )
                        """
                    ]
                    * len(mentions)
                )
                + f"""
                ) matches
                INNER JOIN entities e ON
                    e.id = matches.entity_id AND
                    e.is_meta IS FALSE
                ORDER BY
                    matches.mention
                """,
                list(
                    chain.from_iterable(
                        [mention.text, mention.text, mention.text]
                        for mention in mentions
                    )
                ),
            ).fetchall()
        ]:
            mention = grouped_row.pop("mention")
            grouped_rows[mention] = [*grouped_rows.get(mention, []), grouped_row]

        return grouped_rows

    def _fetch_candidates_by_fts(
        self, mentions: Tuple[Span, ...]
    ) -> Dict[str, List[Dict[str, Union[str, int, float]]]]:
        """Fetches candidates for mentions by searching in Wikidata entity descriptions.
        mentions (Tuple[Span, ...]): List of mentions for which to fetch candidates.
        RETURN (Dict[str, List[Dict[str, Union[str, int, float]]]]): Lists of candidiates per mention, sorted by (1)
            mention and (2) BM25 score.
        """
        # Subquery to fetch alias values for single mentions.
        query = ""
        for i, mention in enumerate(mentions):
            query += f"""
                SELECT
                    '{mention.text}' as mention,
                    match.score,
                    match.entity_id,
                    match.rowid,
                    sum(afe.count) as sum_occurence_count
                FROM (
                    SELECT
                        bm25(entities_texts) as score,
                        et.entity_id,
                        et.ROWID as rowid
                    FROM
                        entities_texts et
                    WHERE
                        entities_texts MATCH '{mention.text}'
                    ORDER BY
                        bm25(entities_texts)
                    LIMIT {self._top_k_entities_fts}
                ) match
                INNER JOIN entities e ON
                    e.ROWID = match.ROWID AND
                    e.is_meta IS FALSE
                INNER JOIN aliases_for_entities afe ON
                    e.id = afe.entity_id
                GROUP BY
                    mention,
                    match.score,
                    match.entity_id,
                    match.rowid
            """
            if i < len(mentions) - 1:
                query += "\nUNION ALL\n"

        grouped_rows: Dict[str, List[Dict[str, Union[str, int, float]]]] = {}
        for row in [dict(row) for row in self._db_conn.execute(query).fetchall()]:
            mention = row.pop("mention")
            grouped_rows[mention] = [*grouped_rows.get(mention, []), row]

        return grouped_rows

    def _init_annoy_from_file(self) -> None:
        """Inits Annoy index."""
        self._annoy = annoy.AnnoyIndex(self.entity_vector_length, "angular")
        self._annoy.load(str(self._paths["annoy"]))

    def _update_hash(self, key: str) -> str:
        """Updates hash.
        key (str): Key for file to hash - has to be in self._paths.
        RETURNS (str): File hash.
        """
        self._hashes[key] = (
            self._hash_file(self._paths[key]) if self._paths[key] is not None else None
        )
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
                        {
                            key: str(path) if path else None
                            for key, path in self._paths.items()
                        },
                        self.entity_vector_length,
                        self._top_k_aliases,
                        self._top_k_entities_alias,
                        self._top_k_entities_fts,
                        self._threshold_alias,
                        self._n_trees,
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
            self._paths = {k: Path(v) if v else None for k, v in meta_info[1].items()}
            self.entity_vector_length = meta_info[2]
            self._top_k_aliases = meta_info[3]
            self._top_k_entities_alias = meta_info[4]
            self._top_k_entities_fts = meta_info[5]
            self._threshold_alias = meta_info[6]
            self._n_trees = meta_info[7]
            self._hashes["db"] = meta_info[8]
            self._hashes["annoy"] = meta_info[9]

            self._db_conn = establish_db_connection(self._language, self._paths["db"])
            self._init_annoy_from_file()

            for file_id in self._hashes:
                assert self._hashes[file_id] == self._hash_file(
                    self._paths[file_id]
                ), f"File with internal ID '{file_id}' does not match deserialized hash."

        spacy.util.from_bytes(
            bytes_data,
            {"meta": deserialize_meta, "vocab": self.vocab.from_bytes},
            exclude,
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

        for file_id in self._paths:
            self._update_hash(file_id)

        serialize = {
            "meta": lambda p: pickle_data(
                (
                    self._language,
                    self._paths,
                    self.entity_vector_length,
                    self._top_k_aliases,
                    self._top_k_entities_alias,
                    self._top_k_entities_fts,
                    self._threshold_alias,
                    self._n_trees,
                    self._hashes,
                ),
                p,
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
                self.entity_vector_length = meta_info[2]
                self._top_k_aliases = meta_info[3]
                self._top_k_entities_alias = meta_info[4]
                self._top_k_entities_fts = meta_info[5]
                self._threshold_alias = meta_info[6]
                self._n_trees = meta_info[7]
                self._hashes = meta_info[8]

                self._db_conn = establish_db_connection(
                    self._language, self._paths["db"]
                )
                self._init_annoy_from_file()

                for file_id in self._hashes:
                    assert self._hashes[file_id] == self._hash_file(
                        self._paths[file_id]
                    ), f"File with internal ID '{file_id}' does not match deserialized hash."

        deserialize = {
            "meta": lambda p: deserialize_meta_info(p),
            "vocab.json": lambda p: self.vocab.strings.from_disk(p),
        }
        spacy.util.from_disk(path, deserialize, exclude)

    @classmethod
    def generate_from_disk(
        cls,
        path: Union[str, Path],
        exclude: Iterable[str] = SimpleFrozenList(),
        artifact_paths: Dict[str, Optional[Path]] = SimpleFrozenDict(),
        check_hashes_of_overridden_artifacts: bool = True,
        **kwargs,
    ) -> "WikiKB":
        """
        Generate WikiKnowledgeBase instance from disk. Passed kwargs override arguments for __init__() of new instance
        read from disk.
        path (Union[str, Path]): Target file path.
        exclude (Iterable[str]): List of components to exclude.
        artifact_paths (Dict[str, Optional[Path]]): Dictionary with paths to external artifacts (such as database or
            index files). Keys not in self._paths are ignored.
        check_hashes_of_overridden_artifacts (bool): Whether to check equality of stored file hashes to the hashes of
            files specified in artifact_paths. It can be useful to disable this for certain workflows - in general this
            should only be set to False if strictly necessary, as it disables a consistency precaution.
            Ideally we support handling of these workflows in a more consistent way and drop this argument.
        return (WikiKB): Generated WikiKB instance.
        """
        path = spacy.util.ensure_path(path)
        if not path.exists():
            raise ValueError(spacy.Errors.E929.format(loc=path))
        if not path.is_dir():
            raise ValueError(spacy.Errors.E928.format(loc=path))
        args: Dict[str, Any] = {"vocab": Vocab(strings=["."])}
        hashes: Dict[str, str] = {}

        def deserialize_meta_info(file_path: Path) -> None:
            """
            Deserializes meta info.
            file_path (Path): File path.
            """
            with open(file_path, "rb") as file:
                meta_info = pickle.load(file)
                args["language"] = meta_info[0]
                args["db_path"] = artifact_paths.get("db", meta_info[1]["db"])
                args["annoy_path"] = artifact_paths.get("annoy", meta_info[1]["annoy"])
                args["entity_vector_length"] = meta_info[2]
                args["top_k_aliases"] = meta_info[3]
                args["top_k_entities_alias"] = meta_info[4]
                args["top_k_entities_fts"] = meta_info[5]
                args["threshold_alias"] = meta_info[6]
                args["n_trees"] = meta_info[7]
                for _file_id, _file_hash in meta_info[8].items():
                    hashes[_file_id] = _file_hash

        spacy.util.from_disk(
            path,
            {
                "meta": lambda p: deserialize_meta_info(p),
                "vocab.json": lambda p: args["vocab"].strings.from_disk(p),
            },
            exclude,
        )

        # Initialize instance, set hashes manually since they aren't specified on initialization.
        kb = cls(**{**args, **kwargs})
        kb._hashes = hashes
        # Check for hash equality (mismatch indicates there might be an issue with DB/Annoy file paths or files).
        for file_id in kb._hashes:
            # Skip hash equality check if file path has been over
            if file_id in artifact_paths and not check_hashes_of_overridden_artifacts:
                continue
            assert kb._hashes[file_id] == kb._hash_file(
                kb._paths[file_id]
            ), f"File with internal ID '{file_id}' does not match deserialized hash."

        return kb

    @staticmethod
    def _hash_file(
        file_path: Optional[Path], blocksize: int = 2**20
    ) -> Optional[str]:
        """Generates MD5 file of hash iteratively (without loading entire file into memory).
        Source: https://stackoverflow.com/a/1131255.
        file_path (Path): Path of file to hash.
        blocksize (int): Size of blocks to load into memory (in bytes).
        RETURN (str): MD5 hash of file. None if file_path is None.
        """
        if file_path is None:
            return None

        file_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read(blocksize)
            while buf:
                file_hash.update(buf)
                buf = f.read(blocksize)
        return file_hash.hexdigest()

    def update_path(self, file_id: str, path: Optional[Path]) -> None:
        """Update path. Includes update of file hash.
        file_id (str): File ID.
        path (Optional[Path]): Path to file.
        """
        self._paths[file_id] = path
        self._update_hash(file_id)
