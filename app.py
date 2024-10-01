import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template
from flask_socketio import SocketIO
from astropy import units as u
from poliastro.earth import EarthSatellite
from poliastro.earth.plotting import GroundtrackPlotter
from sgp4.api import Satrec, jday
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy.time import Time
from poliastro.util import time_range
from astropy.coordinates import SphericalRepresentation
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import threading
import time as time_module
import requests
import logging
from math import sqrt, degrees, radians, cos, sin
from beyond.io.tle import Tle
from beyond.frames import create_station
from beyond.dates import Date
import gc
import os

app = Flask(__name__)
socketio = SocketIO(app)
logging.getLogger('werkzeug').disabled = True

observer_lat = 48.8566
observer_lon = 2.3522
next_passage_data = {}
passage_aos = None
passage_max = None
passage_los = None
tle = None

def GetTLE(norad_cat_id):
    url = f"https://db.satnogs.org/api/tle/?norad_cat_id={norad_cat_id}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        
        if data:
            tle = data[0]
            return tle

def OrbitFromTLE(tle, current_time):
    satellite = Satrec.twoline2rv(tle['tle1'], tle['tle2'])
    jd, fr = jday(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second)
    _, r, v = satellite.sgp4(jd, fr)

    r = np.array(r) * u.km
    v = np.array(v) * u.km / u.s
    orbit = Orbit.from_vectors(Earth, r, v, epoch=Time(current_time))
    
    return orbit

def CalculateVisibilityRadius(altitude_km):
    R_earth_km = 6371.0
    visibility_radius_km = sqrt((R_earth_km + altitude_km)**2 - R_earth_km**2)
    return visibility_radius_km

def GenerateCirclePoints(lat, lon, radius_km, num_points=100):
    circle_lats = []
    circle_lons = []
    for i in range(num_points):
        angle = radians(float(i) / num_points * 360)
        dlat = radius_km / 6371.0 * cos(angle)
        dlon = radius_km / 6371.0 * sin(angle) / cos(radians(lat))
        circle_lats.append(lat + degrees(dlat))
        circle_lons.append(lon + degrees(dlon))
    return circle_lats, circle_lons

def LatLon(orb:Orbit, gp):
    raw_pos, raw_epoch = orb.rv()[0], orb.epoch
    itrs_pos = gp._from_raw_to_ITRS(raw_pos, raw_epoch)
    itrs_latlon_pos = itrs_pos.represent_as(SphericalRepresentation)

    lat = itrs_latlon_pos.lat.to_value(u.deg)
    lon = itrs_latlon_pos.lon.to_value(u.deg)
    return lat, lon

def NextPassage(tle_raw, obs_lat, obs_lon):
    tle = Tle(f"PARISAT\n{tle_raw['tle1']}\n{tle_raw['tle2']}")
    station = create_station('ObserverInput', (obs_lat, obs_lon, 0))
    next_visibility = {'AOS': None, 'MAX': None, 'LOS': None}
    first_passage = None

    for orb in station.visibility(tle.orbit(), start=Date.now(), stop=timedelta(days=1), step=timedelta(minutes=1), events=True):
        azim = -np.degrees(orb.theta) % 360
        elev = np.degrees(orb.phi)
        r = orb.r / 1000.

        first_passage = first_passage or {
            'date':orb.date,
            'azim':azim,
            'elev':elev,
            'dist':r
        }

        if orb.event and orb.event.info == "AOS":
            next_visibility['AOS'] = {
                'date':orb.date,
                'azim':azim,
                'elev':elev,
                'dist':r
            }
            
        if orb.event and orb.event.info == "MAX":
            next_visibility['MAX'] = {
                'date':orb.date,
                'azim':azim,
                'elev':elev,
                'dist':r
            }

        if orb.event and orb.event.info == "LOS":
            next_visibility['LOS'] = {
                'date':orb.date,
                'azim':azim,
                'elev':elev,
                'dist':r
            }
            break
    
    next_visibility['MAX'] = next_visibility['MAX'] or first_passage
    return next_visibility

def NextPassageUpdate(tle, obs_lat, obs_lon):
    next_passage_time = NextPassage(tle, obs_lat, obs_lon)
    global next_passage_data
    azimuth = None
    elevation = None
    distance = None

    azimuth = next_passage_time['MAX']['azim']
    elevation = next_passage_time['MAX']['elev']
    distance = next_passage_time['MAX']['dist']

    next_passage_data = {
        'next_passage_time': (next_passage_time['MAX']['date']).strftime("%Y-%m-%d %H:%M:%S UTC"),
        'azimuth': f"{azimuth:.2f}°",
        'elevation': f"{elevation:.2f}°",
        'distance': f"{distance:.0f} km"
    }        

    return ((next_passage_time['AOS']['date'].datetime.replace(tzinfo=timezone.utc) if next_passage_time.get('AOS') else None),
            (next_passage_time['MAX']['date'].datetime.replace(tzinfo=timezone.utc) if next_passage_time.get('MAX') else None),
            (next_passage_time['LOS']['date'].datetime.replace(tzinfo=timezone.utc) if next_passage_time.get('LOS') else None))

def update_orbit_data():
    global tle, observer_lat, observer_lon, passage_aos, passage_max, passage_los
    tle = GetTLE(98880)
    last_request = datetime.now(timezone.utc).hour
    tle_update_time_iso = tle['updated']
    tle_update_time = datetime.fromisoformat(tle_update_time_iso).strftime('%Y-%m-%d %H:%M:%S UTC')
    passage_aos, passage_max, passage_los = NextPassageUpdate(tle, observer_lat, observer_lon)
    target_iteration_time = 1.0 

    while True:
        iteration_start_time = time_module.time()
        current_time = datetime.now(timezone.utc)
        parisat = OrbitFromTLE(tle, current_time)
        prs_spacecraft = EarthSatellite(parisat, None)
        period_seconds = parisat.period.to(u.second).value
        t_span = time_range(
            current_time, end=current_time + timedelta(seconds=period_seconds)
        )

        gp = GroundtrackPlotter()
        gp.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)'
        )

        gp.update_geos(
            bgcolor='rgba(0,0,0,0)',
            projection_type="natural earth",
            showcountries=True,
            landcolor="#3F83BF",
            oceancolor="#1D4B73",
            countrycolor="rgba(68, 68 68, 0.5)",
            coastlinewidth=0.5,
            countrywidth=0.5,
            lataxis_gridcolor="rgba(128, 128, 128, 0.5)",
            lonaxis_gridcolor="rgba(128, 128, 128, 0.5)",
            lataxis_gridwidth=0.5,
            lonaxis_gridwidth=0.5
        )

        altitude_km = np.linalg.norm(parisat.r.to(u.km).value) - 6371.0
        visibility_radius_km = CalculateVisibilityRadius(altitude_km)
        lat, lon = LatLon(parisat, gp)
        circle_lats, circle_lons = GenerateCirclePoints(lat, lon, visibility_radius_km)

        gp.add_trace(
            go.Scattergeo(
                lon=circle_lons,
                lat=circle_lats,
                mode='lines',
                line=dict(width=0, color='#FF8668'),
                fill='toself',
                fillcolor='rgba(255, 134, 104, 0.5)',
                opacity=0.5
            )
        )

        gp.add_trace(
            go.Scattergeo(
                lat=[observer_lat],
                lon=[observer_lon],
                name="Observer",
                marker={
                    "color": "#ECEFF1",
                    "size": 15,
                    "symbol": "x-thin",
                    "line": {"width": 4, "color": "#ECEFF1"}
                }
            )
        )

        gp.plot(
            prs_spacecraft,
            t_span,
            label="Trajectoire",
            color="#FF3503",
            line_style={"width": 2},
            marker={
                "size": 15,
                "symbol": "circle"
            }
        )
        
        fig_dict = gp.fig.to_dict()
        for trace in fig_dict['data']:
            for key, value in trace.items():
                if isinstance(value, np.ndarray):
                    trace[key] = value.tolist()

        if current_time.second == 0 and current_time.minute == 0 and current_time.hour != last_request:
            tle = GetTLE(98880)
            last_request = current_time.hour
            tle_update_time_iso = tle['updated']
            tle_update_time = datetime.fromisoformat(tle_update_time_iso).strftime('%Y-%m-%d %H:%M:%S')
        
        if current_time > passage_max:
            passage_aos, passage_max, passage_los = NextPassageUpdate(tle, observer_lat, observer_lon)

        socketio.emit('update_orbit', {'fig': fig_dict, 'tle_update_time': tle_update_time})
        socketio.emit('next_passage_time', next_passage_data)
        gc.collect()

        iteration_duration = time_module.time() - iteration_start_time
        sleep_time = max(0, target_iteration_time - iteration_duration)
        time_module.sleep(sleep_time)

@socketio.on('update_observer')
def handle_update_observer(data):
    global tle, observer_lat, observer_lon, next_passage_data, passage_aos, passage_max, passage_los
    observer_lat = data['lat']
    observer_lon = data['lon']
    next_passage_data['next_passage_time'] = 'Calculating...'
    next_passage_data['azimuth'] = None
    next_passage_data['elevation'] = None
    next_passage_data['distance'] = None
    passage_aos, passage_max, passage_los = NextPassageUpdate(tle, observer_lat, observer_lon)

@app.route('/')
def index():
    return render_template('index.html', observer_lat=observer_lat, observer_lon=observer_lon)

if __name__ == '__main__':    
    thread = threading.Thread(target=update_orbit_data)
    thread.daemon = True
    thread.start()
    socketio.run(app, host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=False)