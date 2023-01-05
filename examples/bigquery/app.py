import beam


app = beam.App(
    name="big-query-example",
    cpu=4,
    memory="16Gi",
    python_version="python3.8",
    python_packages=["google-cloud-bigquery", "google-auth", "pandas", "db-dtypes"],
)

# You can run this app on a schedule, every hour
app.Trigger.Schedule(
    when="every 1h",
    # The handler is the function that'll be run when the task is invoked
    handler="run.py:read_from_bq",
)

# We're mounting a file path so we can store our query results somewhere
app.Output.File(path="query_result.csv", name="query_result")
# You can download these files by running this command in your shell: `download <path>`
# The file will be saved to a __downloads__ folder in the same directory where the app is running
# If your app is deployed, you can download these outputs from your Beam dashboard: beam.cloud/apps
