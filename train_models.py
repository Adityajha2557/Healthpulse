"""
Train ML Models - Fixed Version
- Trains on ALL regions and ALL keywords (not just US/3 keywords)
- Fixes: RiskClassifier 'index out of bounds' error
- Fixes: Only 104 samples used (was filtering to single region)
"""

from app.ml_models import OutbreakForecaster, RiskClassifier
from app.database_orchestrator import HealthPulseDatabase
import pandas as pd
import numpy as np
import os
import glob
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ── Patch the broken create_risk_labels method ──────────────────────────────
# Original bug: np.digitize with bins=[25,50,75,100] produces values 0-4,
# then -1 gives -1 to 3, and np.clip(0,3) still leaves -1 possible for 0 values.
# Fix: use pd.cut which is cleaner and always gives exactly 4 buckets.
def _fixed_create_risk_labels(self, df: pd.DataFrame, target_col: str) -> np.ndarray:
    values = df[target_col].values.astype(float)
    mn, mx = values.min(), values.max()
    if mx == mn:
        # All same value → all Low risk
        return np.zeros(len(values), dtype=int)
    normalized = (values - mn) / (mx - mn) * 100
    # 4 equal buckets: 0-25=Low, 25-50=Medium, 50-75=High, 75-100=Critical
    labels = pd.cut(
        normalized,
        bins=[0, 25, 50, 75, 100],
        labels=[0, 1, 2, 3],
        include_lowest=True
    )
    return np.array(labels, dtype=int)

# Apply the patch
RiskClassifier.create_risk_labels = _fixed_create_risk_labels
# ────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 80)
    print("🤖 HEALTHPULSE ML MODEL TRAINING — FULL DATASET")
    print("=" * 80)

    # ── Connect to DB ────────────────────────────────────────────────────────
    db = HealthPulseDatabase(os.getenv('MONGODB_URI'))

    # ── Load ALL data (all regions, all keywords) ────────────────────────────
    print("\n📊 Loading ALL data from MongoDB...")

    # Load for every region we collected
    all_regions = ['US', 'US-CA', 'US-NY', 'US-TX', 'US-FL', 'US-IL']
    all_dfs = []

    for region in all_regions:
        df = db.get_latest_trends(region, days=730)   # 2 years
        if len(df) > 0:
            all_dfs.append(df)
            print(f"   ✅ {region}: {len(df)} records")
        else:
            print(f"   ⚠️  {region}: no data")

    if not all_dfs:
        print("❌ No data loaded at all. Run collect_historical_data.py first.")
        return

    trends_df = pd.concat(all_dfs, ignore_index=True)
    trends_df['date'] = pd.to_datetime(trends_df['date'])

    print(f"\n✅ Total loaded: {len(trends_df):,} records")
    print(f"   Date range : {trends_df['date'].min().date()} → {trends_df['date'].max().date()}")
    print(f"   Keywords   : {sorted(trends_df['keyword'].unique())}")
    print(f"   Regions    : {sorted(trends_df['region'].unique())}")

    os.makedirs('trained_models', exist_ok=True)

    # ========================================================================
    # PART 1: TIME-SERIES FORECASTER — all regions × all keywords
    # ========================================================================
    print("\n" + "=" * 80)
    print("📈 PART 1: TRAINING TIME-SERIES FORECASTERS")
    print("=" * 80)

    forecaster = OutbreakForecaster()
    all_keywords = sorted(trends_df['keyword'].unique())
    forecast_results = []
    skipped = []

    total_combos = len(all_regions) * len(all_keywords)
    trained_count = 0

    for region in all_regions:
        for keyword in all_keywords:
            subset = trends_df[
                (trends_df['keyword'] == keyword) &
                (trends_df['region'] == region)
            ].copy()

            if len(subset) < 30:
                skipped.append(f"{region}/{keyword} ({len(subset)} records)")
                continue

            print(f"\n🔄 [{trained_count+1}/{total_combos}] {region} / {keyword}  ({len(subset)} records)")

            try:
                metrics = forecaster.train(subset, region=region, keyword=keyword)
                model_path = f'trained_models/forecaster_{region.replace("-","_")}_{keyword.replace(" ","_")}.pkl'
                forecaster.save_model(model_path, region, keyword)

                pred = forecaster.predict(region, keyword, periods=14)
                forecast_range = f"{pred['predicted_volume'].min():.1f} – {pred['predicted_volume'].max():.1f}"

                print(f"   ✅ {metrics['model']}  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}")
                print(f"   📅 14-day forecast: {forecast_range}")

                forecast_results.append({
                    'region': region,
                    'keyword': keyword,
                    'model': metrics['model'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'samples': metrics['samples'],
                    'model_path': model_path,
                })
                trained_count += 1

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                skipped.append(f"{region}/{keyword} (error: {e})")

    print(f"\n✅ Forecasters trained: {trained_count} / {total_combos}")
    if skipped:
        print(f"⚠️  Skipped ({len(skipped)}): {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}")

    # ========================================================================
    # PART 2: RISK CLASSIFIER — use ALL region data combined
    # ========================================================================
    print("\n" + "=" * 80)
    print("🎯 PART 2: TRAINING RISK CLASSIFIER (ALL REGIONS)")
    print("=" * 80)

    print("\n🔄 Engineering features from ALL trend data...")

    # Aggregate by date + region
    features_df = trends_df.groupby(['date', 'region']).agg(
        volume_mean=('search_volume', 'mean'),
        volume_std=('search_volume', 'std'),
        volume_max=('search_volume', 'max'),
        volume_min=('search_volume', 'min'),
        growth_mean=('growth_rate', 'mean'),
        growth_std=('growth_rate', 'std'),
        zscore_mean=('z_score', 'mean'),
        zscore_max=('z_score', 'max'),
        anomaly_count=('is_anomaly', 'sum'),
        keyword_count=('keyword', 'nunique'),
    ).reset_index()

    features_df = features_df.fillna(0)

    # Rolling features per region
    features_df = features_df.sort_values(['region', 'date'])
    features_df['volume_7d_avg'] = features_df.groupby('region')['volume_mean'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    features_df['volume_14d_avg'] = features_df.groupby('region')['volume_mean'].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    features_df['volume_momentum'] = features_df['volume_7d_avg'] - features_df['volume_14d_avg']

    print(f"✅ Feature matrix: {len(features_df):,} samples × {len(features_df.columns)} columns")
    print(f"   Regions covered: {sorted(features_df['region'].unique())}")

    feature_cols = [
        'volume_mean', 'volume_std', 'volume_max', 'volume_min',
        'growth_mean', 'growth_std', 'zscore_mean', 'zscore_max',
        'anomaly_count', 'volume_7d_avg', 'volume_14d_avg', 'volume_momentum',
        'keyword_count',
    ]

    X = features_df[feature_cols]
    classifier = RiskClassifier(model_type='xgboost')
    y = classifier.create_risk_labels(features_df, 'volume_mean')

    print(f"\n🔄 Training XGBoost classifier...")
    print(f"   Total samples : {len(X):,}")
    print(f"   Features      : {len(feature_cols)}")

    unique, counts = np.unique(y, return_counts=True)
    print(f"   Risk distribution:")
    for level, count in zip(unique, counts):
        label = RiskClassifier.RISK_LABELS[level] if level < len(RiskClassifier.RISK_LABELS) else f"class_{level}"
        print(f"      {label:10s}: {count:4d}  ({count/len(y)*100:.1f}%)")

    # Check we have all 4 classes — if not, XGBoost num_class will mismatch
    n_classes = len(np.unique(y))
    print(f"   Distinct classes in labels: {n_classes}")

    try:
        metrics = classifier.train(X, y)

        print(f"\n✅ CLASSIFIER RESULTS:")
        print(f"   Accuracy       : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"   Cross-val score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        print(f"   Training samples: {metrics['train_samples']:,}")
        print(f"   Test samples    : {metrics['test_samples']:,}")

        classifier.save_model('trained_models/risk_classifier.pkl')
        print("   ✅ Saved: trained_models/risk_classifier.pkl")

        # Quick sanity test
        sample = X.sample(5, random_state=42)
        preds = classifier.predict_risk_score(sample)
        print(f"\n   Sample predictions:")
        print(preds[['risk_score', 'risk_label', 'confidence']].to_string(index=False))

    except Exception as e:
        print(f"❌ Classifier training failed: {e}")
        import traceback; traceback.print_exc()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)

    models = glob.glob('trained_models/*.pkl')
    total_size_mb = sum(os.path.getsize(p) for p in models) / 1024 / 1024

    print(f"\n✅ Models saved: {len(models)} files  ({total_size_mb:.1f} MB total)")
    for p in sorted(models):
        kb = os.path.getsize(p) / 1024
        print(f"   {os.path.basename(p):55s} {kb:8.1f} KB")

    if forecast_results:
        results_df = pd.DataFrame(forecast_results)
        print(f"\n📈 Forecaster performance summary:")
        print(results_df.groupby('model')[['rmse', 'mae']].mean().round(2).to_string())

    print("\n" + "=" * 80)
    print("✅ FULL MODEL TRAINING COMPLETE!")
    print("\nNext steps:")
    print("   1. Start API:  uvicorn app.main:app --reload --port 8000")
    print("   2. Test:       http://localhost:8000/docs")
    print("   3. Deploy:     follow the deployment guide")
    print("=" * 80)

    db.close()


if __name__ == "__main__":
    main()
