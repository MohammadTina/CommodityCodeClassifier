"""
Query builder for fetching commodity codes with full hierarchy.
Builds the joined query across all active hierarchy tables.
All table and column names are driven by config — no hardcoding.
"""

import logging
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class QueryBuilder:
    def __init__(self, config: dict, engine: Engine):
        self.config = config
        self.engine = engine
        self.tables = config.get("tables", {})
        self.columns = config.get("columns", {})
        self.active_filter = config.get("active_filter", "VALID_TO IS NULL")

    def fetch_all_national_codes(self) -> list[dict]:
        """
        Fetch all active national commodity codes with full hierarchy context.
        This is the data that gets embedded and indexed.

        Returns a list of dicts, one per national code, containing:
        - national_code (constructed from HS6_COD + TAR_PR1)
        - All description fields from each hierarchy level
        - Legal notes from section and chapter levels
        """
        query = self._build_hierarchy_query()
        logger.info("Fetching all active national commodity codes...")

        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()

        records = [dict(zip(columns, row)) for row in rows]
        logger.info(f"Fetched {len(records)} active national codes")
        return records

    def _build_hierarchy_query(self) -> str:
        """
        Build the full hierarchy join query dynamically from config.
        Joins: national → subheading → heading → chapter → section
        """
        nat = self.tables["national"]
        sub = self.tables["subheading"]
        hdg = self.tables["heading"]
        chp = self.tables["chapter"]
        sec = self.tables["section"]

        # column references
        nat_col = self.columns["national"]
        sub_col = self.columns["subheading"]
        hdg_col = self.columns["heading"]
        chp_col = self.columns["chapter"]
        sec_col = self.columns["section"]

        query = f"""
            SELECT
                -- National code (constructed)
                n.{nat_col['hs6_code']} || n.{nat_col['pr1']} AS national_code,

                -- National level descriptions
                n.{nat_col['description']}      AS tar_dsc,
                n.{nat_col['description_long']} AS tar_all,

                -- Subheading level
                h6.{sub_col['code']}            AS hs6_cod,
                h6.{sub_col['description']}     AS hs6_dsc,

                -- Heading level
                h4.{hdg_col['code']}            AS hs4_cod,
                h4.{hdg_col['description']}     AS hs4_dsc,

                -- Chapter level
                h2.{chp_col['code']}            AS hs2_cod,
                h2.{chp_col['description']}     AS hs2_dsc,
                h2.{chp_col['notes']}           AS hs2_txt,

                -- Section level
                h1.{sec_col['code']}            AS hs1_cod,
                h1.{sec_col['description']}     AS hs1_dsc,
                h1.{sec_col['notes']}           AS hs1_txt

            FROM {nat} n

            JOIN {sub} h6
                ON n.{nat_col['hs6_code']} = h6.{sub_col['code']}
                AND (h6.{self.columns['subheading']['valid_to']} IS NULL)

            JOIN {hdg} h4
                ON h6.{sub_col['parent_code']} = h4.{hdg_col['code']}
                AND (h4.{self.columns['heading']['valid_to']} IS NULL)

            JOIN {chp} h2
                ON h4.{hdg_col['parent_code']} = h2.{chp_col['code']}
                AND (h2.{self.columns['chapter']['valid_to']} IS NULL)

            JOIN {sec} h1
                ON h2.{chp_col['parent_code']} = h1.{sec_col['code']}
                AND (h1.{self.columns['section']['valid_to']} IS NULL)

            WHERE n.{nat_col['valid_to']} IS NULL
            and n.tar_pr1 not like '%*%'
        """
        return query

    def fetch_by_national_code(self, national_code: str) -> dict | None:
        """
        Fetch a single national code with full hierarchy.
        Used for validation and result enrichment.
        """
        # split national code back into HS6 + PR1
        hs6_code = national_code[:6]
        pr1 = national_code[6:]

        nat = self.tables["national"]
        nat_col = self.columns["national"]

        base_query = self._build_hierarchy_query()
        full_query = f"""
            SELECT * FROM ({base_query}) sub
            WHERE sub.hs6_cod = :hs6_code
            AND sub.national_code = :national_code
        """

        with self.engine.connect() as conn:
            result = conn.execute(
                text(full_query),
                {"hs6_code": hs6_code, "national_code": national_code}
            )
            row = result.fetchone()

        if row:
            return dict(zip(result.keys(), row))
        return None
