import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

plotly_template = 'plotly_dark'

def trend_chart(hist_df, forecast_df, zones):
    df_h = hist_df[hist_df['zone'].isin(zones)].copy()
    df_f = forecast_df[forecast_df['zone'].isin(zones)].copy()
    
    df_h['Type'] = 'Historical'
    df_f['Type'] = 'Forecast'
    
    combined = pd.concat([df_h, df_f])
    fig = px.line(combined, x='date', y='cases', color='zone', line_dash='Type',
                  title='Influenza Cases: Historical & Forecast', template=plotly_template)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def resource_bar_chart(alloc_df):
    melted = alloc_df.melt(id_vars=['zone'], value_vars=['alloc_icu_beds', 'alloc_doctors', 'alloc_oxygen_units'])
    fig = px.bar(melted, x='zone', y='value', color='variable', barmode='group',
                 title='Resource Allocation per Zone', template=plotly_template,
                 labels={'value': 'Count', 'variable': 'Resource Type'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def urgency_heatmap(urgency_df):
    df_heat = urgency_df[['zone', 'pred_cases', 'growth_rate', 'urgency_score']].set_index('zone')
    fig = px.imshow(df_heat, text_auto=True, color_continuous_scale='Reds', aspect='auto',
                    title='Zone Risk Metrics Heatmap', template=plotly_template)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def xai_waterfall(xai):
    waterfall_data = xai['waterfall']
    fig = go.Figure(go.Waterfall(
        name="Urgency", orientation="v",
        measure=["relative"] * len(waterfall_data) + ["total"],
        x=list(waterfall_data.keys()) + ["Total Score"],
        y=list(waterfall_data.values()) + [xai['urgency_score']],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="Urgency Score Breakdown", template=plotly_template,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def feature_importance_chart(feat_imp, zone):
    df = feat_imp[feat_imp['zone'] == zone].sort_values('importance', ascending=True)
    fig = px.bar(df, x='importance', y='feature', orientation='h',
                 title=f'Model Features Impact for {zone}', template=plotly_template)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def kpi_summary(urgency_df, hist_df):
    return {
        "latest_daily": int(hist_df[hist_df['date'] == hist_df['date'].max()]['cases'].sum()),
        "total_pred_7d": int(urgency_df['pred_cases'].sum() * 7),
        "high_risk_zones": len(urgency_df[urgency_df['urgency_score'] > 0.5]),
        "avg_growth_pct": float(urgency_df['growth_rate'].mean() * 100),
        "total_icu": int(urgency_df['icu_capacity'].sum())
    }
