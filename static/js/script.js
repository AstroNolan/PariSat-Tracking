document.addEventListener('DOMContentLoaded', (event) => {
    var socket = io();

    var config = {
        staticPlot: true,
        responsive: true
    };

    socket.on('update_orbit', function(data) {
        Plotly.react('plot', data.fig.data, data.fig.layout, config);
        document.getElementById('tle-update').innerText = 'TLE Update: ' + data.tle_update_time;
    });

    socket.on('next_passage_time', function(data) {
        document.getElementById('next-passage-time').innerText = data.next_passage_time;
        document.getElementById('next-passage-azimuth').innerText = data.azimuth ? 'Azimut : ' + data.azimuth : '';
        document.getElementById('next-passage-elevation').innerText = data.elevation ? 'Élévation : ' + data.elevation : '';
        document.getElementById('next-passage-distance').innerText = data.distance ? 'Distance : ' + data.distance : '';
    });

    document.getElementById('update-observer').addEventListener('click', function() {
        var lat = parseFloat(document.getElementById('latitude').value);
        var lon = parseFloat(document.getElementById('longitude').value);
        console.log(`Emitting update_observer with Latitude = ${lat}, Longitude = ${lon}`);
        socket.emit('update_observer', { lat: lat, lon: lon });
    });
});