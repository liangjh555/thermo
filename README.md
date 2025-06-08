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
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('lake_prediction.log')
    ]
)
logger = logging.getLogger(__name__)

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
                'n_estimators': 200,
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
        f"未找到匹配的TIFF文件: 目录={tif_dir}, 变量={var_prefix}, 年份={year}, 场景={scenario}")

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
                
                if np.any(mask_valid)
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
        logger.info("\n开始训练随机森林模型")
        perf = {}
        
        for cls in ['a', 'b', 'c', 'd', 'e', 'f']:
            subset = data[data['class'] == cls]
            n = len(subset)
            
            if n < 5:
                logger.warning(f"类别 {cls} 样本 {n} < 5，跳过训练")
                continue
            
            logger.info(f"训练类别 {cls} (样本数={n})")
            X = subset[self.features].values
            y = subset['area'].values

            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)
            self.scalers[cls] = scaler

            model = self.models[cls]
            model.fit(Xs, y)

            y_pred = model.predict(Xs)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            min_area = float(subset['area'].min())
            max_area = float(subset['area'].max())

            perf[cls] = {
                'n_lakes': n, 
                'min_area': min_area, 
                'max_area': max_area, 
                'r2': r2, 
                'mae': mae, 
                'rmse': rmse
            }
            
            self.training_history.append({
                'class': cls, 
                'n_samples': n, 
                'r2': r2, 
                'mae': mae, 
                'rmse': rmse,
                'min_area': min_area, 
                'max_area': max_area, 
                'timestamp': datetime.now()
            })
            
            logger.info(f"类别 {cls} 结果 - R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

            if self.config['output']['save_models']:
                model_path = os.path.join(self.results_dir, f"model_{cls}.pkl")
                scaler_path = os.path.join(self.results_dir, f"scaler_{cls}.pkl")
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                logger.info(f"模型保存至: {model_path}, {scaler_path}")
        
        perf_df = pd.DataFrame(perf).T
        perf_df = perf_df[['n_lakes', 'min_area', 'max_area', 'r2', 'mae', 'rmse']]
        
        perf_path = os.path.join(self.results_dir, 'model_performance.csv')
        perf_df.to_csv(perf_path)
        logger.info(f"模型性能保存至: {perf_path}")
        
        return perf_df

    def predict_for_lake(self, climate_row: pd.Series, lake_class: str):
        """预测单个湖泊面积"""
        if lake_class not in self.models or lake_class not in self.scalers:
            return np.nan
        
        try:
            X_raw = climate_row[self.features].values.reshape(1, -1)
            X_scaled = self.scalers[lake_class].transform(X_raw)
            return float(self.models[lake_class].predict(X_scaled)[0])
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return np.nan

    def predict_for_dataset(self, base_data: pd.DataFrame, climate_df: pd.DataFrame) -> pd.Series:
        """并行预测整个数据集"""
        def predict_single(lake_id, row):
            if lake_id in climate_df.index:
                try:
                    return self.predict_for_lake(climate_df.loc[lake_id], row['class'])
                except Exception as e:
                    logger.error(f"湖泊 {lake_id} 预测失败: {str(e)}")
                    return np.nan
            return np.nan
        
        results = Parallel(n_jobs=self.config['model_params']['n_jobs'])(
            delayed(predict_single)(lake_id, row)
            for lake_id, row in base_data.iterrows()
        )
        
        return pd.Series(results, index=base_data.index)

    def validate(self, base_data: pd.DataFrame, validation_data: pd.DataFrame):
        """模型验证"""
        if validation_data is None:
            logger.info("无验证数据，跳过验证")
            return None
        
        common_ids = base_data.index.intersection(validation_data.index)
        if not common_ids.size:
            logger.warning("基础数据和验证数据没有共同的lake_id，无法验证")
            return None
        
        logger.info(f"验证模型，共同湖泊数: {len(common_ids)}")
        sub_base = base_data.loc[common_ids]
        sub_val = validation_data.loc[common_ids]
        
        pred_areas = self.predict_for_dataset(sub_base, sub_val)
        true_areas = sub_val['area_observed']
        
        r2 = r2_score(true_areas, pred_areas)
        mae = mean_absolute_error(true_areas, pred_areas)
        rmse = np.sqrt(mean_squared_error(true_areas, pred_areas))
        corr, p_val = pearsonr(true_areas, pred_areas)
        
        logger.info(f"验证结果 - R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}, Corr={corr:.3f}")
        
        class_metrics = {}
        for cls in ['a', 'b', 'c', 'd', 'e', 'f']:
            cls_mask = (sub_base['class'] == cls)
            if cls_mask.sum() > 0:
                t = true_areas[cls_mask]
                p = pred_areas[cls_mask]
                
                class_metrics[cls] = {
                    'n': t.size,
                    'r2': r2_score(t, p),
                    'mae': mean_absolute_error(t, p),
                    'rmse': np.sqrt(mean_squared_error(t, p))
                }
                logger.info(f"类别 {cls} 验证 - R²={class_metrics[cls]['r2']:.3f}")
                metrics_data = {
            'overall': {
                'n_lakes': len(common_ids),
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'corr': corr,
                'p_value': p_val
            }
        }
        
        for cls, metrics in class_metrics.items():
            metrics_data[f'class_{cls}'] = metrics
        
        metrics_df = pd.DataFrame(metrics_data).T
        
        val_path = os.path.join(self.results_dir, 'validation_metrics.csv')
        metrics_df.to_csv(val_path)
        logger.info(f"验证指标保存至: {val_path}")
        
        self._plot_validation(true_areas, pred_areas, sub_base['class'])
        
        return metrics_df

    def _plot_validation(self, true, pred, classes):
        """绘制验证结果图"""
        try:
            plt.figure(figsize=(10, 8))
            sc = plt.scatter(true, pred, c=classes.factorize()[0], cmap='viridis', alpha=0.7)
            plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', linewidth=1)
            plt.xlabel('观测面积 (ha)')
            plt.ylabel('预测面积 (ha)')
            plt.title('观测 vs 预测')
            plt.colorbar(sc, label='湖泊类别')
            plt.grid(linestyle='--', alpha=0.5)
            plt.savefig(os.path.join(self.results_dir, 'validation_scatter.png'), dpi=300)
            plt.close()
            
            residuals = pred - true
            plt.figure(figsize=(10, 6))
            plt.scatter(true, residuals, c=classes.factorize()[0], cmap='viridis', alpha=0.7)
            plt.axhline(0, color='r', linestyle='--')
            plt.xlabel('观测面积 (ha)')
            plt.ylabel('残差 (预测 - 观测)')
            plt.title('预测残差')
            plt.grid(linestyle='--', alpha=0.5)
            plt.savefig(os.path.join(self.results_dir, 'validation_residuals.png'), dpi=300)
            plt.close()
            
            logger.info("验证可视化图已保存")
        except Exception as e:
            logger.error(f"生成验证图失败: {str(e)}")
    
    def plot_future_tla(self, tla_df):
        try:
            plt.figure(figsize=(12, 7))
            
            scenarios = [col for col in tla_df.columns if col != 'year']
            
            for scenario in scenarios:
                plt.plot(tla_df['year'], tla_df[scenario], label=scenario, linewidth=2.5)
            
            plt.title('2020-2100年热喀斯特湖总面积预测', fontsize=14)
            plt.xlabel('年份', fontsize=12)
            plt.ylabel('总面积 (ha)', fontsize=12)
            plt.legend(title='情景')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.axvline(2020, color='gray', linestyle='--', alpha=0.7)
            plt.text(2020.5, plt.ylim()[1]*0.9, '2020年', rotation=90, fontsize=10)
            
            plot_path = os.path.join(self.results_dir, 'future_tla_plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"未来TLA预测图保存至: {plot_path}")
        except Exception as e:
            logger.error(f"生成未来TLA图失败: {str(e)}")

logger.info("\n初始化模型")
predictor = LakeAreaPredictor(config)

logger.info("\n训练模型")
perf_df = predictor.train(lake_df)

if validation_df is not None:
    logger.info("\n验证模型")
    val_metrics = predictor.validate(lake_df, validation_df)

logger.info("\n开始未来TLA预测")

def predict_future_tla(predictor, base_data, future_df, years, scenarios):
    """预测未来热喀斯特湖总面积"""
    tla_results = {sc: [] for sc in scenarios}
    
    for sc in tqdm(scenarios, desc="情景进度"):
        scenario_data = future_df[future_df['scenario'] == sc].set_index('lake_id')
        
        for yr in tqdm(years, desc=f"{sc} 年份进度", leave=False):

            year_data = scenario_data[scenario_data['year'] == yr]
            
            pred_areas = predictor.predict_for_dataset(base_data, year_data)
            
            total_tla = pred_areas.sum(skipna=True)
            tla_results[sc].append(total_tla)
    
    return tla_results

future_tla = predict_future_tla(predictor, lake_df, future_df_all, future_years, future_scenarios)

tla_df = pd.DataFrame({'year': future_years})
for sc in future_scenarios:
    tla_df[sc] = future_tla[sc]

tla_path = os.path.join(config['output']['results_dir'], 'tla_predictions.csv')
tla_df.to_csv(tla_path, index=False)
logger.info(f"未来TLA预测结果保存至: {tla_path}")

predictor.plot_future_tla(tla_df)

logger.info("\n分析完成!")
