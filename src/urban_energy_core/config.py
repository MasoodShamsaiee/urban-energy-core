from pathlib import Path


DATA_REPO_NAME = "urban-energy-data"
ELEC_4CITIES_FILE = "Donnees_4villes_RES.parquet"
ELEC_RAW_SUBDIR = "electricity"
DA_ELEC_FILE = "Donnees_DA.parquet"
LOCAL_TZ = "America/Toronto"
CENSUS_FSA_SUBDIR = "census/FSA scale"
CENSUS_DA_SUBDIR = "census/DA scale"
GEOMETRY_RAW_SUBDIR = "geometry"
MONTREAL_FSA_GEOJSON = "Montreal.geojson"
MONTREAL_DA_GEOJSON = "Montreal_DA.geojson"
QUEBEC_CITY_FSA_GEOJSON = "Quebec_city.geojson"
QUEBEC_CITY_DA_GEOJSON = "Quebec_city_DA.geojson"
TROIS_RIVIERES_FSA_GEOJSON = "Trois_Rivieres.geojson"
TROIS_RIVIERES_DA_GEOJSON = "Trois_Rivieres_DA.geojson"
WEATHER_RAW_SUBDIR = "weather"
MONTREAL_WEATHER_FILE = "weather_montreal.csv"
QUEBEC_CITY_WEATHER_FILE = "weather_quebec.csv"
TROIS_RIVIERES_WEATHER_FILE = "weather_trois_rivieres.csv"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_repo_root() -> Path:
    return repo_root().parent / DATA_REPO_NAME


def default_data_dir() -> Path:
    return default_data_repo_root() / "data"
