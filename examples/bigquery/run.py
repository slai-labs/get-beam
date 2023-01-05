import os
import io
from google.cloud import bigquery
from google.oauth2 import service_account

# To start, you'll need your own GCP credentials stored in the Beam Secrets Manager
# beam.cloud/apps/settings/secrets


class BigQueryClient:
    def __init__(self):
        # Load credentials from the secrets manager
        self.credentials = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )

        # Load the project ID
        self.client = bigquery.Client(
            project=os.environ["BQ_PROJECT_ID"],
        )

    def run_query(self, query, output_path):
        # Pull from BQ and convert to dataframe
        dataframe = self.client.query(query).result().to_dataframe()
        # Save DF as CSV and write it to the Output Dir defined in app.py
        dataframe.to_csv(
            output_path,
            mode="w+",
        )


def read_from_bq(*, query):
    bq_client = BigQueryClient()
    # We'll download the query results to the Output directory mounted in app.py
    # You can download files in your shell by running: `download query_result.csv`
    # The file will be saved to a __downloads__ folder in the same directory where the app is running
    output_path = "query_result.csv"
    bq_client.run_query(query=query, output_path=output_path)


if __name__ == "__main__":
    query = """
    SELECT
    refresh_date AS Day,
    term AS Top_Term,
        -- These search terms are in the top 25 in the US each day.
    rank,
    FROM `bigquery-public-data.google_trends.top_terms`
    WHERE
    rank = 1
        -- Choose only the top term each day.
    AND refresh_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 WEEK)
        -- Filter to the last 2 weeks.
    GROUP BY Day, Top_Term, rank
    ORDER BY Day DESC
    """
    read_from_bq(query=query)