import os
import influxdb_client
import pandas as pd
# Ignore "Missing pivot function" warning
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)

from config import CONFIG


#######################################
# GLOBAL VARIABLES
#######################################


INFLUX_BUCKET = None
INFLUX_CLIENT = None

CHANNEL_ALIAS = {
    'counts_10': 'counts, No Wt, <10 (Primary)',
    'counts_10-50': 'counts, No Wt, 10-50 (Primary)',
    'counts_50-150': 'counts, No Wt, 50-150 (Primary)',
    'counts_150-300': 'counts, No Wt, 150-300 (Primary)',
    'counts_300-1000': 'counts, No Wt, 300-1000 (Primary)',
    'counts_1000': 'counts, No Wt, >1000 (Primary)'
}

SUBSTANCE_ALIAS = {
    'aa': 'adipic_acid',
    'pdp': 'kdp',
}


#######################################
# INFLUXDB
#######################################


def init_influx():
    global INFLUX_BUCKET, INFLUX_CLIENT

    influx_url = os.environ.get("INFLUX_URL")
    if not influx_url:
        influx_url = os.environ.get("INFLUXDB_URL")
    influx_token = os.environ.get("INFLUX_TOKEN")
    if not influx_token:
        influx_token = os.environ.get("INFLUXDB_TOKEN")
    INFLUX_BUCKET = CONFIG.get("influx", {}).get("influx_bucket")
    influx_org = CONFIG.get("influx", {}).get("influx_org")

    INFLUX_CLIENT = influxdb_client.InfluxDBClient(
        url=influx_url,
        org=influx_org,
        token=influx_token,
        verify_ssl=False  # Disable, if cert verification fails
    )


def get_experiment_timeseries(experiment_id: str | int, device: str, channel: str, substance: str) -> pd.DataFrame:
    if not isinstance(experiment_id, str) and not isinstance(experiment_id, int):
        raise TypeError(f'Expected type int or str for experiment_id, got: {type(experiment_id)}.')
    if not isinstance(device, str):
        raise TypeError(f'Expected type str for device, got: {type(device)}.')
    if not isinstance(channel, str):
        raise TypeError(f'Expected type str for channel, got: {type(channel)}.')
    if not isinstance(substance, str):
        raise TypeError(f'Expected type str for substance, got: {type(substance)}.')
    if not INFLUX_CLIENT or not INFLUX_BUCKET:
        init_influx()

    # Some features specify channel names different from what the timeseries is actually named in influxdb
    channel_alias = CHANNEL_ALIAS.get(channel, "not-a-real-channel")
    substance_alias = SUBSTANCE_ALIAS.get(substance, "not-a-real-substance")

    query = f'''
        from(bucket:"{INFLUX_BUCKET}")
        |> range(start: 2020-10-19T00:00:00Z)
        |> filter(fn:(r) => r.experiment_id == "{experiment_id}")
        |> filter(fn: (r) => r._measurement == "{device}")
        |> filter(fn: (r) => r.substance == "{substance}" or r.substance == "{substance_alias}")
        |> filter(fn: (r) => r._field == "{channel}" or r._field == "{channel_alias}")
        |> keep(columns: ["_time", "_value"])
    '''
    query_api = INFLUX_CLIENT.query_api()
    df_data = query_api.query_data_frame(query)
    if df_data is None or (isinstance(df_data, pd.DataFrame) and df_data.empty):
        raise RuntimeError(f'No data for experiment "{experiment_id}", device "{device}" and channel "{channel}" was returned by InfluxDB.')
    if '_time' not in df_data.columns or '_value' not in df_data.columns:
        raise ValueError(f'No _time column or no _value column found in data returned from InfluxDB.')
    df_data = df_data.loc[:, ["_time", "_value"]]
    df_data = df_data.rename(columns={'_value': f'{device}:{channel}'})
    df_data = df_data.set_index('_time')
    return df_data


# Init influxdb on import
init_influx()
