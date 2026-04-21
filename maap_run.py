from maap.maap import MAAP

maap = MAAP(maap_host="api.maap-project.org")

# Follow https://github.com/zhouqiang06/hls-composite/tree/main for algo logic
jobs = []
for tile in ["14VLQ", "18WXS", "16WFB", "26WMC", "19VDL"]:
    job = maap.submitJob(
        algo_id="HLSpheno",
        version="v0.1",
        identifier="test-run",
        queue="maap-dps-worker-8gb",
        tile=tile,
        target_year="2020",
        out_dir="s3://maap-ops-workspace/shared/zhouqiang06/HLS_composite",
    )
    jobs.append(job)
