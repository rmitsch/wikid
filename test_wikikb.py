import os
import pickle
import tempfile
from pathlib import Path
from typing import List

import pytest
import spacy
from spacy.tokens import Span, Doc

from src.kb import WikiKB
from src.extraction import establish_db_connection

_language = "en"


@pytest.fixture
def _db_path() -> Path:
    """Generates test DB.
    RETURNS (Path): Path to database in temporary directory.
    """
    tmp_dir = Path(tempfile.TemporaryDirectory().name)
    db_path = tmp_dir / "wiki.sqlite"

    # Construct DB.
    db_conn = establish_db_connection(_language, db_path)
    with open(
        Path(os.path.abspath(__file__)).parent / "src" / "extraction" / "ddl.sql",
        "r",
    ) as ddl_sql:
        db_conn.cursor().executescript(ddl_sql.read())

    # Fill DB.
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO entities (id) VALUES ('Q60'), ('Q100'), ('Q597');")
    cursor.execute(
        """
        INSERT INTO entities_texts (entity_id, name, description, label) VALUES
            ('Q60', 'New York City', 'most populous city in the United States', 'New York City'),
            ('Q100', 'Boston', 'capital and largest city of Massachusetts, United States', 'Boston'),
            ('Q597', 'Lisbon', 'capital city of Portugal', 'Lisbon'),
            (
                'Q131371', 'Boston Celtics', 'NBA team based in Boston; tied with most NBA Championships',
                'Boston Celtics'
            ),
            (
                'Q131364', 'New York Knicks', 'National Basketball Association franchise in New York City',
                'New York Knicks'
            );
        """
    )
    cursor.execute(
        """
        INSERT INTO articles (entity_id, id) VALUES
            ('Q60', 0), ('Q100', 1), ('Q597', 2), ('Q131371', 3), ('Q131364', 4)
        ;
        """
    )
    cursor.execute(
        """
        INSERT INTO articles_texts (entity_id, title, content) VALUES
            (
                'Q60',
                'New York City',
                'New York, often called New York City (NYC), is the most populous city in the United States. With a
                2020 population of 8,804,190 distributed over 300.46 square miles (778.2 km2), New York City is also the
                most densely populated major city in the United States. The city is within the southern tip of New York
                State, and constitutes the geographical and demographic center of both the Northeast megalopolis and the
                New York metropolitan area – the largest metropolitan area in the world by urban landmass.'
            ),
            (
                'Q100',
                'Boston',
                'Boston (US: /ˈbɔːstən/), officially the City of Boston, is the state capital and most populous city of
                the Commonwealth of Massachusetts, as well as the cultural and financial center of the New England
                region of the United States. It is the 24th-most populous city in the country. The city boundaries
                encompass an area of about 48.4 sq mi (125 km2) and a population of 675,647 as of 2020.'
            ),
            (
                'Q597',
                'Lisbon',
                'Lisbon (/ˈlɪzbən/; Portuguese: Lisboa [liʒˈboɐ] (listen)) is the capital and the largest city of
                Portugal, with an estimated population of 544,851 within its administrative limits in an area of
                100.05 km2. Lisbon''s urban area extends beyond the city''s administrative limits with a population of
                around 2.7 million people, being the 11th-most populous urban area in the European Union. About 3
                million people live in the Lisbon metropolitan area, making it the third largest metropolitan area in
                the Iberian Peninsula, after Madrid and Barcelona.'
            ),
            (
                'Q131371',
                'Boston Celtics',
                'The Boston Celtics (/ˈsɛltɪks/ SEL-tiks) are an American professional basketball team based in Boston.
                The Celtics compete in the National Basketball Association (NBA) as a member of the league''s Eastern
                Conference Atlantic Division. Founded in 1946 as one of the league''s original eight teams, the Celtics
                play their home games at TD Garden, which they share with the National Hockey League''s Boston Bruins.
                The Celtics are one of the most successful basketball teams in NBA history. The franchise is one of two
                teams with 17 NBA Championships, the other franchise being the Los Angeles Lakers. The Celtics currently
                hold the record for the most recorded wins of any NBA team.'
            ),
            (
                'Q131364',
                'New York Knicks',
                'The New York Knickerbockers, shortened and more commonly referred to as the New York Knicks, are an
                American professional basketball team based in the New York City borough of Manhattan. The Knicks
                compete in the National Basketball Association (NBA) as a member of the Atlantic Division of the Eastern
                Conference. The team plays its home games at Madison Square Garden, an arena they share with the New
                York Rangers of the National Hockey League (NHL). They are one of two NBA teams located in New York
                City; the other team is the Brooklyn Nets. Alongside the Boston Celtics, the Knicks are one of two
                original NBA teams still located in its original city.'
            )
        ;
        """
    )

    cursor.execute(
        """
        INSERT INTO aliases_for_entities (alias, entity_id, count, prior_prob) VALUES
            ('NYC', 'Q60', 1, 0.01),
            ('New York', 'Q60', 1, 0.01),
            ('the five boroughs', 'Q60', 1, 0.01),
            ('Big Apple', 'Q60', 1, 0.01),
            ('City of New York', 'Q60', 1, 0.01),
            ('NY City', 'Q60', 1, 0.01),
            ('New York, New York', 'Q60', 1, 0.01),
            ('New York City, New York', 'Q60', 1, 0.01),
            ('New York, NY', 'Q60', 1, 0.01),
            ('New York City (NYC)', 'Q60', 1, 0.01),
            ('New York (city)', 'Q60', 1, 0.01),
            ('New York City, NY', 'Q60', 1, 0.01),
            ('Caput Mundi', 'Q60', 1, 0.01),
            ('The City So Nice They Named It Twice', 'Q60', 1, 0.01),
            ('Capital of the World', 'Q60', 1, 0.01),

            ('Boston', 'Q100', 1, 0.01),
            ('Beantown', 'Q100', 1, 0.01),
            ('The Cradle of Liberty', 'Q100', 1, 0.01),
            ('The Hub', 'Q100', 1, 0.01),
            ('The Cradle of Modern America', 'Q100', 1, 0.01),
            ('The Athens of America', 'Q100', 1, 0.01),
            ('The Walking City', 'Q100', 1, 0.01),
            ('The Hub of the Universe', 'Q100', 1, 0.01),
            ('Bostonia', 'Q100', 1, 0.01),
            ('Boston, Massachusetts', 'Q100', 1, 0.01),
            ('Boston, Mass.', 'Q100', 1, 0.01),
            ('Puritan City', 'Q100', 1, 0.01),
            ('Lisbon', 'Q597', 1, 0.01),
            ('Lisboa', 'Q597', 1, 0.01),
            ('Boston Celtics', 'Q131371', 1, 0.01),
            ('Celtics', 'Q131371', 1, 0.01),
            ('Celts', 'Q131371', 1, 0.01),
            ('C''s', 'Q131371', 1, 0.01),
            ('Green and White', 'Q131371', 1, 0.01),
            ('New York Knicks', 'Q131364', 1, 0.01),
            ('New York Knickerbockers', 'Q131364', 1, 0.01),
            ('Knicks', 'Q131364', 1, 0.01),
            ('NYK', 'Q131364', 1, 0.01),
            ('Minutemen', 'Q131364', 1, 0.01)
            ;
        """
    )
    cursor.execute(
        "INSERT INTO aliases (word) SELECT distinct(alias) FROM aliases_for_entities;"
    )
    db_conn.commit()
    return db_path


@pytest.fixture
def _kb(_db_path) -> WikiKB:
    """Generates KB.
    _db_path (Path): Path to database / fixture constructing database in temporary directory.
    RETURNS (WikiKB): WikiKB instance.
    """
    nlp = spacy.load("en_core_web_sm")
    kb = WikiKB(
        nlp.vocab,
        nlp(".").vector.shape[0],
        _db_path,
        _db_path.parent / "wiki.annoy",
        "en",
        use_coref=True,
    )
    kb.build_embeddings_index(nlp, n_jobs=1)

    return kb


@pytest.fixture
def _kb_with_lookup_file(_kb, _db_path, _doc_with_ents) -> WikiKB:
    """
    Generates WikiKB using a lookup file.
    _kb (WikiKB): KB without lookup file, used to generated lookup file.
    _db_path (_db_path): Path to generated database.
    RETURNS (WikiKB): WikiKB using a lookup file.
    """
    cands = list(next(_kb.get_candidates_all([_doc_with_ents])))
    lookup_path = _db_path.parent / "mention_lookups.pkl"

    with open(lookup_path, "wb") as file:
        pickle.dump(
            {mention.text: cands[i] for i, mention in enumerate(_doc_with_ents.ents)},
            file,
        )

    kb = WikiKB(
        _kb.vocab,
        _kb.entity_vector_length,
        _db_path,
        _db_path.parent / "wiki.annoy",
        "en",
        mentions_candidates_path=lookup_path,
    )
    # Skip embeddings index generation since it already exists (via fixture _kb).

    return kb


@pytest.fixture
def _doc_with_ents() -> Doc:
    """
    Returns a test Doc instance with defined .ents.
    RETURNS (Doc): Doc instance with defined .ents.
    """
    doc = spacy.load("en_core_web_sm")("new yorc and Boston")
    doc.ents = [
        Span(doc, 0, 2, kb_id="Q60", label="NIL"),
        Span(doc, 3, 4, kb_id="Q100", label="NIL"),
    ]
    return doc


def test_initialization(_kb) -> None:
    """Tests KB intialization."""
    # Check DB content.
    assert all(
        [
            _kb._db_conn.cursor()
            .execute(f"SELECT count(*) FROM {table}")
            .fetchone()["count(*)"]
            == 3
            for table in ("entities", "articles", "entities_texts", "articles_texts")
        ]
    )
    assert len(_kb) == 3
    assert (
        _kb._db_conn.cursor()
        .execute("SELECT count(*) FROM aliases_for_entities")
        .fetchone()["count(*)"]
        == 29
    )
    assert (
        _kb._db_conn.cursor()
        .execute("SELECT count(*) FROM aliases")
        .fetchone()["count(*)"]
        == 29
    )

    # Check Annoy index.
    assert len(_kb._annoy.get_item_vector(0)) == _kb.entity_vector_length
    assert _kb._annoy.get_n_items() == 3


@pytest.mark.parametrize("method", ["bytes", "disk"])
def test_serialization(_kb, method: str) -> None:
    """Tests KB serialization (to and from byte strings, to and from disk).
    method (str): Method to use for serialization. Has to be one of ("bytes", "disk").
    """
    assert method in ("bytes", "disk")
    nlp = spacy.load(
        "en_core_web_sm", exclude=["tagger", "lemmatizer", "attribute_ruler"]
    )

    # Create KB for comparison with diverging values.
    kb = WikiKB(
        nlp.vocab,
        nlp(".").vector.shape[0] + 1,
        _kb._paths["db"],
        Path("this_path_doesnt_exist"),
        "es",
        n_trees=100,
        top_k_aliases=100,
        top_k_entities_alias=100,
        top_k_entities_fts=100,
        threshold_alias=1000,
    )

    # Reset KB to serialized reference KB.
    if method == "bytes":
        kb.from_bytes(_kb.to_bytes())
    else:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            kb_file_path = Path(tmp_dir_name) / "kb"
            _kb.to_disk(kb_file_path)
            kb.from_disk(kb_file_path)

    assert _verify_kb_equality(_kb, kb)


def test_factory_method(_kb) -> None:
    """Tests factory method to generate WikiKB instance from file."""
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        kb_file_path = Path(tmp_dir_name) / "kb"
        _kb.to_disk(kb_file_path)
        kb = WikiKB.generate_from_disk(kb_file_path)

    assert _verify_kb_equality(_kb, kb)


def _verify_kb_equality(kb1: WikiKB, kb2: WikiKB) -> bool:
    """Checks whether kb1 and kb2 have identical values for all arguments (doesn't check on DB equality).
    kb1 (WikiKB): First instance.
    kb2 (WikiKB): Second instance.
    RETURNS (bool): Whether kb1 and kb2 have identical values for all arguments.
    """
    return all(
        [
            getattr(kb1, attr_name) == getattr(kb2, attr_name)
            for attr_name in (
                "_paths",
                "_language",
                "_n_trees",
                "entity_vector_length",
                "_hashes",
                "_top_k_aliases",
                "_top_k_entities_alias",
                "_top_k_entities_fts",
                "_threshold_alias",
                "_use_coref",
            )
        ]
    )


def _verify_candidate_retrieval_results(
    kb: WikiKB, doc: Doc, target_entity_ids: List[List[str]]
):
    """Assert that retrieved candidates are correct.
    kb (WikiKB): KB to use.
    doc (Doc): Doc with .ents to resolve.
    target_entity_ids (List[List[str]]): Expected target entity IDs per mention.
    """
    for i, (cands_from_all, cands_from_single) in enumerate(
        zip(
            next(kb.get_candidates_all([doc])),
            [kb.get_candidates(mention) for mention in doc.ents],
        )
    ):
        assert len(list(cands_from_all)) == len(list(cands_from_single))
        for j, (cand_all, cand_single) in enumerate(
            zip(cands_from_all, cands_from_single)
        ):
            assert cand_all.entity == target_entity_ids[i][j]
            # Check for equality between candidates generated by get_candidates_all() and those generated by
            # get_candidates().
            for prop in ("entity", "entity_", "entity_vector", "mention", "prior_prob"):
                assert getattr(cand_all, prop) == getattr(cand_single, prop)


def test_get_candidates(_kb, _doc_with_ents) -> None:
    """Smoke test to guarantee equivalency between get_candidates() and get_candidates_all()."""
    _verify_candidate_retrieval_results(_kb, _doc_with_ents, [["Q60"], ["Q100", "Q60"]])


def test_serialized_mention_lookups(_kb_with_lookup_file, _doc_with_ents) -> None:
    """Tests serialized mention lookups."""
    # Monkeypatch candidate search methods to make sure we don't use them.
    def _raise(_) -> None:
        raise Exception

    _kb_with_lookup_file._fetch_candidates_by_alias = _raise

    # Test lookup works correctly.
    _verify_candidate_retrieval_results(
        _kb_with_lookup_file, _doc_with_ents, [["Q60"], ["Q100", "Q60"]]
    )


def test_coref(_kb):
    doc = spacy.load("en_core_web_sm")(
        "The Boston Celtics played against the New York Knicks today. Boston beat New York by 5 points. The game took "
        "place in New York instead of in Boston."
    )
    doc.ents = [
        Span(doc, 1, 3, kb_id="Q131364", label="NIL"),  # Boston Celtics
        Span(doc, 6, 9, kb_id="Q131371", label="NIL"),  # New York Knicks
        Span(doc, 11, 12, kb_id="Q131371", label="NIL"),  # Boston
        Span(doc, 13, 15, kb_id="Q60", label="NIL"),  # New York,
        Span(doc, 24, 26, kb_id="Q131371", label="NIL"),  # New York
        Span(doc, 29, 30, kb_id="Q60", label="NIL"),  # Boston
    ]

    for i, cands in enumerate(next(_kb.get_candidates_all([doc.ents_spangroup]))):
        pass
