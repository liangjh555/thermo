import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import geopandas as gpd 
import rasterio 
from rasterio.mask import mask 
from shapely.geometry import mapping 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import pearsonr 
from tqdm import tqdm 
import joblib 
import yaml 
from datetime import datetime 
from joblib import Parallel, delayed 
import logging 

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[logging.StreamHandler(), logging.FileHandler('lake_prediction.log')]
) 
logger = logging.getLogger("lake_predictor")


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        default_config = {
            'data_paths': {
                'lake_polygons': 'lakes.gpkg',
                'validation_lake_polygons': 'validation_lakes.gpkg',
                'base_tif_dir': 'base_data_2020',
                'future_tif_dir': 'future_climate',
                'validation_tif_dir': 'validation_data_2024'
            },
            'tif_variables': {
                'temperature': 'temp',
                'ground_temp': 'gst',
                'precipitation': 'precip',
                'evapotranspiration': 'et',
                'alt': 'alt'
            },
            'tif_naming': {
                'separator': '_',
                'order': ['scenario', 'variable', 'year']
            },
            'model_params': {
                'n_estimators': 500,
                'max_depth': 5,
                'random_state': 42,
                'n_jobs': -1
            },
            'raster_processing': {
                'crs': 'EPSG:32649',
                'buffer_distance': 100
            },
            'output': {
                'results_dir': 'results',
                'save_models': True
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        logger.info(f"创建默认配置文件: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    required_sections = ['data_paths', 'tif_variables', 'model_params', 'raster_processing', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必要部分: {section}")

    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    return config


config = load_config()
logger.info("配置加载完成")


def load_lake_polygons(file_path, config):
    """加载湖泊矢量数据并进行预处理"""
    logger.info(f"加载湖泊矢量: {file_path}")

    if file_path.endswith('.gpkg'):
        import fiona
        gdf = gpd.read_file(file_path, layer='lakes') if 'lakes' in fiona.listlayers(file_path) else gpd.read_file(file_path)
    elif file_path.endswith(('.shp', '.geojson')):
        gdf = gpd.read_file(file_path)
    else:
        raise ValueError(f"不支持的矢量文件格式: {file_path}")

    required_cols = ['lake_id', 'class']
    for col in required_cols:
        if col not in gdf.columns:
            raise ValueError(f"湖泊矢量缺少必要列: {col}")

    gdf = gdf.set_index('lake_id')

    target_crs = config['raster_processing']['crs']
    if gdf.crs is None:
        logger.warning("矢量无 CRS，假设 EPSG:4326")
        gdf.crs = 'EPSG:4326'

    if gdf.crs.to_string() != target_crs:
        logger.info(f"投影到 {target_crs}")
        gdf = gdf.to_crs(target_crs)

    gdf['area'] = gdf.geometry.area

    buffer_dist = config['raster_processing']['buffer_distance']
    logger.info(f"创建 {buffer_dist} 米缓冲区")
    gdf['buffer_geom'] = gdf.geometry.buffer(buffer_dist)

    gdf['centroid'] = gdf.geometry.centroid

    return gdf


def find_matching_tif(tif_dir, var_prefix, year, scenario):
    """查找匹配的TIFF文件"""
    naming = config.get('tif_naming', {})
    separator = naming.get('separator', '_')
    order = naming.get('order', ['scenario', 'variable', 'year'])

    parts = {
        'scenario': scenario,
        'variable': var_prefix,
        'year': str(year)
    }

    possible_names = [
        separator.join([parts[k] for k in order]) + '.tif',
        separator.join([parts[k].lower() for k in order]) + '.tif',
        separator.join([parts[k].upper() for k in order]) + '.tif'
    ]

    for fname in possible_names:
        tif_path = os.path.join(tif_dir, fname)
        if os.path.exists(tif_path):
            return tif_path

    all_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
    matching_files = []

    for f in all_files:
        if (var_prefix in f and str(year) in f and 
            (scenario is None or scenario in f)):
            matching_files.append(f)

    if matching_files:
        return os.path.join(tif_dir, sorted(matching_files)[0])

    raise FileNotFoundError(
        f"未找到匹配的TIFF文件: 目录={tif_dir}, 变量={var_prefix}, 年份={year}, 场景={scenario}"
    )


def extract_raster_values(gdf, tif_dir, var_prefix, year, scenario=None):
    """从栅格中提取湖泊区域的值"""
    try:
        tif_path = find_matching_tif(tif_dir, var_prefix, year, scenario)
        logger.info(f"处理栅格: {os.path.basename(tif_path)}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return pd.Series(np.nan, index=gdf.index)

    values = []
    with rasterio.open(tif_path) as src:
        if src.crs.to_string() != gdf.crs.to_string():
            gdf_ = gdf.to_crs(src.crs)
            logger.info(f"转换湖泊数据坐标系到: {src.crs.to_string()}")
        else:
            gdf_ = gdf
        
        for lake_id, row in tqdm(gdf_.iterrows(), total=len(gdf_), desc=f"提取 {var_prefix}"):
            val = np.nan
            try:
                geom = row['buffer_geom']
                
                out_img, _ = mask(
                    src, 
                    [mapping(geom)], 
                    crop=True, 
                    nodata=src.nodata,
                    all_touched=True 
                )
                
                band = out_img[0]
                mask_valid = band != src.nodata
                
                if np.any(mask_valid):
                    val = float(np.nanmean(band[mask_valid]))
                else:
                    centroid = row['centroid']
                    py, px = src.index(centroid.x, centroid.y)
                    
                    window = ((py-1, py+2), (px-1, px+2))
                    try:
                        window_data = src.read(1, window=window, boundless=True, fill_value=src.nodata)
                        valid_pixels = window_data[window_data != src.nodata]
                        if valid_pixels.size > 0:
                            val = float(np.nanmean(valid_pixels))
                        else:
                            val = np.nan
                    except Exception as e:
                        logger.warning(f"湖泊 {lake_id} 窗口读取失败: {str(e)}")
                        val = np.nan
            except Exception as e:
                logger.error(f"湖泊 {lake_id} 提取失败: {str(e)}")
                val = np.nan
            
            values.append(val)

    return pd.Series(values, index=gdf.index)


def load_and_process_data(config):
    """加载和处理所有必要数据"""
    data = {}
    paths = config['data_paths']
    var_prefixes = config['tif_variables']

    lake_gdf = load_lake_polygons(paths['lake_polygons'], config)
    data['lake_gdf'] = lake_gdf

    logger.info("\n提取2020年基础数据:")
    for var_name, prefix in var_prefixes.items():
        logger.info(f"处理变量: {var_name}")
        series_2020 = extract_raster_values(
            lake_gdf,
            paths['base_tif_dir'],
            prefix,
            year=2020,
            scenario='base'
        )
        lake_gdf[var_name] = series_2020

    logger.info("\n准备未来气候数据框架...")

    all_files = [f for f in os.listdir(paths['future_tif_dir']) if f.endswith('.tif')]
    years = set()
    scenarios = set()

    for f in all_files:
        parts = f.split('.')[0].split('_')  
        if len(parts) >= 3:
            if config['tif_naming']['order'][-1] == 'year':
                year = parts[-1]
                scenario = '_'.join(parts[:-2]) 
            else:
                year = parts[2] 
                scenario = parts[0] 
            
            if year.isdigit():
                years.add(int(year))
                scenarios.add(scenario)

    years = sorted(years)
    scenarios = sorted(scenarios)

    logger.info(f"检测到年份: {min(years)}-{max(years)}")
    logger.info(f"检测到情景: {', '.join(scenarios)}")

    future_records = []
    for scenario in scenarios:
        for year in years:
            for lake_id in lake_gdf.index:
                future_records.append({
                    'lake_id': lake_id,
                    'year': year,
                    'scenario': scenario
                })

    future_df = pd.DataFrame(future_records)

    logger.info("\n提取未来气候数据:")
    for var_name, prefix in tqdm(var_prefixes.items(), desc="变量进度"):
        var_values = []
        
        for scenario in tqdm(scenarios, desc=f"情景进度 - {var_name}", leave=False):
            for year in tqdm(years, desc=f"年份进度 - {scenario}", leave=False):
                s = extract_raster_values(
                    lake_gdf, 
                    paths['future_tif_dir'], 
                    prefix, 
                    year=year, 
                    scenario=scenario
                )
                
                for lake_id, val in s.items():
                    var_values.append({
                        'lake_id': lake_id,
                        'year': year,
                        'scenario': scenario,
                        var_name: val
                    })
        
        var_df = pd.DataFrame(var_values)
        future_df = future_df.merge(var_df, how='left', on=['lake_id', 'year', 'scenario'])

    data['future_df'] = future_df

    val_dir = paths.get('validation_tif_dir')
    val_lake_path = paths.get('validation_lake_polygons')

    if val_dir and os.path.exists(val_dir):
        logger.info("\n处理验证数据:")
        
        if val_lake_path and os.path.exists(val_lake_path):
            val_lake_gdf = load_lake_polygons(val_lake_path, config)
            logger.info(f"使用验证年份的湖泊矢量: {val_lake_path}")
        else:
            logger.warning("未找到验证湖泊矢量，使用2020年湖泊数据")
            val_lake_gdf = lake_gdf.copy()
        
        val_files = [f for f in os.listdir(val_dir) if f.endswith('.tif')]
        val_years = set()
        
        for f in val_files:
            parts = f.split('.')[0].split('_')
            if len(parts) >= 1 and parts[-1].isdigit():
                val_years.add(int(parts[-1]))
        
        if val_years:
            val_year = max(val_years)
            logger.info(f"使用 {val_year} 年作为验证年份")
            
            val_df = pd.DataFrame(index=val_lake_gdf.index)
            val_df.index.name = 'lake_id'
            
            for var_name, prefix in var_prefixes.items():
                s = extract_raster_values(
                    val_lake_gdf, 
                    val_dir, 
                    prefix, 
                    year=val_year, 
                    scenario='val'
                )
                val_df[var_name] = s
            
            val_df['area_observed'] = val_lake_gdf['area']
            
            data['validation_df'] = val_df
            data['validation_year'] = val_year
        else:
            logger.warning("未检测到验证年份，跳过验证")
            data['validation_df'] = None
    else:
        logger.info("无验证目录，跳过验证")
        data['validation_df'] = None

    return data, years, scenarios


data, future_years, future_scenarios = load_and_process_data(config) 
lake_gdf = data['lake_gdf'] 
future_df_all = data['future_df'] 
validation_df = data.get('validation_df')

lake_df = lake_gdf[['class', 'area'] + list(config['tif_variables'].keys())].copy()

if validation_df is not None: 
    validation_df = validation_df.reset_index().set_index('lake_id')


class LakeAreaPredictor:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.features = list(config['tif_variables'].keys())

        for cls in ['a', 'b', 'c', 'd', 'e', 'f']:
            p = config['model_params']
            self.models[cls] = RandomForestRegressor(
                n_estimators=p['n_estimators'],
                max_depth=p['max_depth'],
                random_state=p['random_state'],
                n_jobs=p['n_jobs']
            )
        
        self.training_history = []
        self.results_dir = config['output']['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)

    def train(self, data: pd.DataFrame):
        logger.info("\n开始训练随机森林模型（70%训练, 30%测试 + 交叉验证）")
        perf = {}
        
        for cls in ['a', 'b', 'c', 'd', 'e', 'f']:
            subset = data[data['class'] == cls]
            n = len(subset)
            
            if n < 20:
                logger.warning(f"类别 {cls} 样本 {n} < 20，跳过训练（样本过少）")
                continue
            
            logger.info(f"训练类别 {cls} (样本数={n})")
            X = subset[self.features].values
            y = subset['area'].values

            # === 数据标准化 ===
            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)
            self.scalers[cls] = scaler

            # === 数据划分 (70% 训练, 30% 测试) ===
            X_train, X_test, y_train, y_test = train_test_split(
                Xs, y, test_size=0.3, random_state=self.config['model_params']['random_state']
            )

            # === 模型初始化 ===
            p = self.config['model_params']
            model = RandomForestRegressor(
                n_estimators=p['n_estimators'],
                max_depth=p['max_depth'],
                random_state=p['random_state'],
                n_jobs=p['n_jobs']
            )

            # === 模型训练 ===
            model.fit(X_train, y_train)
            self.models[cls] = model

            # === 训练集性能 ===
            y_train_pred = model.predict(X_train)
            r2_train = r2_score(y_train, y_train_pred)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

            # === 测试集性能 ===
            y_test_pred = model.predict(X_test)
            r2_test = r2_score(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # === 5折交叉验证 ===
            cv_scores = cross_val_score(model, Xs, y, cv=5, scoring='r2')
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

            # === 保存性能结果 ===
            perf[cls] = {
                'n_lakes': n,
                'train_r2': r2_train, 'train_mae': mae_train, 'train_rmse': rmse_train,
                'test_r2': r2_test, 'test_mae': mae_test, 'test_rmse': rmse_test,
                'cv_r2_mean': cv_mean, 'cv_r2_std': cv_std
            }

            self.training_history.append({
                'class': cls, 'n_samples': n,
                'train_r2': r2_train, 'test_r2': r2_test,
                'cv_r2_mean': cv_mean, 'cv_r2_std': cv_std,
                'timestamp': datetime.now()
            })

            logger.info(
                f"类别 {cls} -> 训练集 R²={r2_train:.3f}, 测试集 R²={r2_test:.3f}, "
                f"CV R²={cv_mean:.3f}±{cv_std:.3f}"
            )

            # === 保存模型与 scaler ===
            if self.config['output']['save_models']:
                model_path = os.path.join(self.results_dir, f"model_{cls}.pkl")
                scaler_path = os.path.join(self.results_dir, f"scaler_{cls}.pkl")
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)

        # === 输出汇总表 ===
        perf_df = pd.DataFrame(perf).T
        perf_df.to_csv(os.path.join(self.results_dir, 'model_performance.csv'))
        logger.info("训练完成，性能指标已保存")

        return perf_df

    # 其余 predict_for_lake / predict_for_dataset / validate / _plot_validation / plot_future_tla 方法保持不变
    # ...（略，和您原始代码一致，这里就不全部重复了）


logger.info("\n初始化预测器")
predictor = LakeAreaPredictor(config)

logger.info("\n训练模型")
perf_df = predictor.train(lake_df)
logger.info(f"\n训练完成，性能汇总:\n{perf_df}")

