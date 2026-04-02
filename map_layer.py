import pydeck as pdk
from data_generator import ZONES

def build_map(urgency_df):
    map_data = []
    for _, row in urgency_df.iterrows():
        zone = row['zone']
        meta = ZONES[zone]
        
        score = row['urgency_score']
        if score > 0.5:
            color = [248, 113, 113, 200]  # Red
        elif score > 0.3:
            color = [251, 191, 36, 200]   # Yellow
        else:
            color = [74, 222, 128, 200]   # Green
            
        map_data.append({
            "zone": zone,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "urgency": f"{score:.3f}",
            "radius": max(1500, score * 6000), 
            "color": color
        })
        
    layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
    )
    
    view_state = pdk.ViewState(
        latitude=13.0600,
        longitude=80.2450,
        zoom=10.5,
        pitch=45,
    )
    
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{zone}\nUrgency: {urgency}"},
        map_style="mapbox://styles/mapbox/dark-v11"
    )
