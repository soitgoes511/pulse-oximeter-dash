import os
from datetime import timedelta
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
from plotly.subplots import make_subplots
from fbprophet import Prophet

# Load .env variables for API
load_dotenv()

TOKEN = os.getenv("TOKEN")
ORG = os.getenv("ORG")
URL = os.getenv("CONNECTION")

client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
query_api = client.query_api()

df = query_api.query_data_frame(
    'from(bucket: "pulse_oximeter") '
    "|> range(start: -1h) "
    "|> filter(fn: (r) => "
    '  r._measurement == "spo2") '
    "|> window(every: 10s) "
    "|> mean()"
)


def sanitize_dataframe(input_df):
    """Function is used to sanitize dataframe by rounding values
    converting timestamp string to an actual datetime, converting timezone
    from UTC time to CET, and finally pivoting the df to create a column
    for both the SPO2 and BPM values.

    :input_df contains the returned df from influx/flux query
    :df_pivot returns the sanitized dataframe used to generate plots
    """
    input_df.dropna(inplace=True)
    input_df["_value"] = input_df["_value"].round(decimals=2)
    input_df["_start"] = pd.to_datetime(input_df["_start"])
    input_df["_start"] = (
        input_df["_start"].dt.tz_convert("Europe/Oslo").dt.tz_localize(None)
    )
    input_df.drop(columns=["table", "_measurement", "result"], inplace=True)
    df_pivot = input_df.pivot(
        index=["_start", "_stop"], columns="_field", values="_value"
    )
    df_pivot.reset_index(inplace=True)
    df_pivot["_stop"] = pd.to_datetime(input_df["_stop"])
    df_pivot["_stop"] = (
        df_pivot["_stop"].dt.tz_convert("Europe/Oslo").dt.tz_localize(None)
    )

    return df_pivot


def create_current_sats_plot(df_current):
    """Function creates 2 y-axis plot using plotly. The left
    y-axis (primary) displays the magnitude of the spo2 percentage
    and the right y-axis (secondary) displays the magnitude of
    the bpm.

    :df input needs changed since it is global but contains data
    :fig is the left side plot of the current sats
    """
    try:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=df_current["_time"],
                y=df_current["spo2"],
                line=dict(color="#6fc3df", width=2, shape="spline"),
                mode="lines",
                name="SPO2",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df_current["_time"],
                y=[92] * len(df_current["_time"]),
                line=dict(color="#ffe64d", width=2, dash="dashdot"),
                name="SPO2 LL",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df_current["_time"],
                y=df_current["bpm"],
                line=dict(color="#df740c", width=2, shape="spline"),
                mode="lines",
                name="BPM",
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=df_current["_time"],
                y=[170] * len(df_current["_time"]),
                line=dict(color="#ff410d", width=2, dash="dashdot"),
                name="BPM UL",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            plot_bgcolor="#171b26",
            paper_bgcolor="#171b26",
            uirevision="dropdown-time-window",
            width=1000,
            height=450,
            margin=dict(t=50),
            hovermode="x",
            font=dict(size=20, color="#fff",),
            xaxis=dict(
                zeroline=False,
                automargin=True,
                showticklabels=True,
                linecolor="#737a8d",
                gridcolor="#737a8d",
                tickfont=dict(color="#fff"),
            ),
            yaxis=dict(
                automargin=True,
                showticklabels=True,
                tickfont=dict(color="#fff"),
                gridcolor="#171b26",
                range=[(df_current["spo2"].min() - 1), (df_current["spo2"].max() + 1)],
            ),
            yaxis2=dict(
                automargin=True,
                showticklabels=True,
                tickfont=dict(color="#fff"),
                gridcolor="#171b26",
                range=[(df_current["bpm"].min() - 1), (df_current["bpm"].max() + 1)],
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            title={
                "text": "SPO2(%) & BPM: Time Series",
                "y": 0.95,
                "x": 0.05,
                "xanchor": "left",
                "yanchor": "top",
            },
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="SPO2(%)", secondary_y=False)
        fig.update_yaxes(title_text="BPM", secondary_y=True)
    except KeyError:
        fig = go.Figure()
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "DISCONNECTED",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 28},
                }
            ],
        )

    return fig


def make_histogram(df_hist, value):
    """Function constructs histogram if user selects this from dropdown"""
    if value == "SPO2H":
        hist_color = "#6fc3df"
    else:
        hist_color = "#df740c"

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_hist["y"], nbinsx=30, marker_color=hist_color))
    fig.update_layout(
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        width=1000,
        height=450,
        margin=dict(t=50),
        hovermode="x",
        bargap=0.1,
        font=dict(size=20, color="#fff",),
        xaxis=dict(
            zeroline=False,
            automargin=True,
            showticklabels=True,
            linecolor="#737a8d",
            gridcolor="#737a8d",
            tickfont=dict(color="#fff"),
        ),
        yaxis=dict(
            automargin=True,
            showticklabels=True,
            tickfont=dict(color="#fff"),
            gridcolor="#171b26",
        ),
    )

    return fig


def reconstruct_forecast(raw_data, forecast):
    """Function rebuilds fbprophet forecast for aesthetics"""
    raw_data.set_index("ds", inplace=True)
    forecast.set_index("ds", inplace=True)
    viz_df = raw_data.join(forecast[["yhat", "yhat_lower", "yhat_upper"]], how="outer")
    last_date = raw_data.index[-1]
    last_date = last_date - timedelta(days=1)
    mask = viz_df.index > last_date
    predict_df = viz_df.loc[mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=raw_data.index,
            y=raw_data["y"],
            line=dict(color="#6fc3df", width=3),
            mode="markers",
            name="ACTUAL",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predict_df.index,
            y=predict_df["yhat"],
            line=dict(color="#d8dae7", width=3),
            name="YHAT",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predict_df.index,
            y=predict_df["yhat_upper"],
            line=dict(color="#df740c"),
            fill="tonexty",
            name="YHAT_HIGH",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predict_df.index,
            y=predict_df["yhat_lower"],
            line=dict(color="#df740c"),
            fill="tonexty",
            name="YHAT_LOW",
        )
    )
    fig.update_layout(
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        width=1000,
        height=450,
        margin=dict(t=50),
        hovermode="x",
        uirevision="dropdown-forecast",
        xaxis=dict(
            zeroline=False,
            automargin=True,
            showticklabels=True,
            linecolor="#737a8d",
            gridcolor="#737a8d",
            tickfont=dict(color="#fff"),
            rangeselector=dict(
                bgcolor="#171b26",
                buttons=list(
                    [
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=3, label="3d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="todate"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all"),
                    ]
                ),
            ),
            # rangeslider=dict(
            #    visible=True
            # ),
            type="date",
        ),
        yaxis=dict(
            automargin=True,
            showticklabels=True,
            tickfont=dict(color="#fff"),
            gridcolor="#171b26",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=18, color="#fff",),
    )

    return fig


def reconstruct_components(forecast):
    """Function reconstruct component plot"""
    forecast.set_index("ds", inplace=True)
    last_read = forecast.index[-1]
    # weekly = last_read - timedelta(days=7)
    daily = last_read - timedelta(days=1)
    # week_mask = (forecast.index > weekly)
    day_mask = forecast.index > daily
    # weekly_df = forecast.loc[week_mask]
    daily_df = forecast.loc[day_mask]
    fig = make_subplots(rows=2, cols=1)
    fig.append_trace(
        go.Scatter(
            x=forecast.index, y=forecast["trend"], line=dict(color="#6fc3df", width=3),
        ),
        row=1,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            x=daily_df.index, y=daily_df["daily"], line=dict(color="#6fc3df", width=3),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        width=1000,
        height=450,
        margin=dict(t=50),
        hovermode="x",
        showlegend=False,
        xaxis=dict(
            zeroline=True,
            automargin=True,
            showticklabels=True,
            linecolor="#737a8d",
            gridcolor="#737a8d",
            tickfont=dict(color="#fff"),
        ),
        yaxis=dict(
            automargin=True,
            showticklabels=True,
            tickfont=dict(color="#fff"),
            gridcolor="#171b26",
        ),
        font=dict(size=18, color="#fff"),
    )
    fig.update_xaxes(gridcolor="#737a8d", linecolor="#737a8d", row=2, col=1)
    fig.update_yaxes(gridcolor="#171b26", row=2, col=1)
    fig.update_xaxes(gridcolor="#737a8d", linecolor="#737a8d", row=3, col=1)
    fig.update_yaxes(gridcolor="#171b26", row=3, col=1)

    return fig


def make_forecast(value):
    """Function input is dropdown selection of user to select right chart"""
    df_forecast = query_api.query_data_frame(
        'from(bucket: "downsampled_pulse_ox")'
        "|> range(start: -7d)"
        '|> filter(fn: (r) => r._measurement == "spo2")'
        "|> aggregateWindow(every: 30m, fn: mean, createEmpty: false)"
        '|> drop(columns: ["_start","_stop","_measurement","uid"])'
        '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
    )

    client.__del__()

    df_forecast["_time"] = pd.to_datetime(df_forecast["_time"])
    df_forecast["_time"] = (
        df_forecast["_time"].dt.tz_convert("Europe/Oslo").dt.tz_localize(None)
    )
    df_forecast.dropna(inplace=True)

    if value == "SPO2H":
        df_forecast.drop(columns=["bpm", "perf_index"], inplace=True)
        df_forecast.rename(columns={"_time": "ds", "spo2": "y"}, inplace=True)
        fig = make_histogram(df_forecast, value)

        return fig

    elif value == "BPMH":
        df_forecast.drop(columns=["spo2", "perf_index"], inplace=True)
        df_forecast.rename(columns={"_time": "ds", "bpm": "y"}, inplace=True)
        fig = make_histogram(df_forecast, value)

        return fig

    elif value == "SPO2":
        df_forecast.drop(columns=["bpm", "perf_index"], inplace=True)
        df_forecast.rename(columns={"_time": "ds", "spo2": "y"}, inplace=True)
        df_forecast["cap"] = 100.0

        my_model = Prophet(interval_width=0.95)
        my_model.fit(df_forecast)
        future_dates = my_model.make_future_dataframe(periods=144, freq="30min")
        future_dates["cap"] = 100.0
        forecast = my_model.predict(future_dates)
        fig = reconstruct_forecast(df_forecast, forecast)

        return fig

    elif value == "SPO2C":
        df_forecast.drop(columns=["bpm", "perf_index"], inplace=True)
        df_forecast.rename(columns={"_time": "ds", "spo2": "y"}, inplace=True)
        df_forecast["cap"] = 100.0

        my_model = Prophet(interval_width=0.95)
        my_model.fit(df_forecast)
        future_dates = my_model.make_future_dataframe(periods=144, freq="30min")
        future_dates["cap"] = 100.0
        forecast = my_model.predict(future_dates)
        fig = reconstruct_components(forecast)

        return fig

    elif value == "BPM":
        df_forecast.drop(columns=["spo2", "perf_index"], inplace=True)
        df_forecast.rename(columns={"_time": "ds", "bpm": "y"}, inplace=True)

        my_model = Prophet(interval_width=0.95)
        my_model.fit(df_forecast)
        future_dates = my_model.make_future_dataframe(periods=144, freq="30min")
        forecast = my_model.predict(future_dates)
        fig = reconstruct_forecast(df_forecast, forecast)

        return fig

    else:
        df_forecast.drop(columns=["spo2", "perf_index"], inplace=True)
        df_forecast.rename(columns={"_time": "ds", "bpm": "y"}, inplace=True)

        my_model = Prophet(interval_width=0.95)
        my_model.fit(df_forecast)
        future_dates = my_model.make_future_dataframe(periods=144, freq="30min")
        forecast = my_model.predict(future_dates)
        fig = reconstruct_components(forecast)

        return fig


app = dash.Dash(__name__, update_title=None)

server = app.server
app.title = "Pulse-Oximeter: Time-Series"


def build_header():
    """Function creates web app header"""
    return html.Div(
        children=[
            html.P("Melody's Dashboard"),
            html.Div(
                children=[
                    html.Button("Export CSV", id="export-csv-button"),
                    Download(id="download-csv"),
                    html.Button("Export XLSX", id="export-xlsx-button"),
                    Download(id="download-xlsx"),
                ],
                className="export-options",
            ),
        ],
        className="header",
    )


def build_footer():
    """Function creates web app footer"""
    return html.Div(
        children=[
            html.P("-MH- 2020"),
            html.P("This app is dedicated to my precious baby girl"),
        ],
        className="footer",
    )


def build_main():
    """Function contains all layout plots and leds in between head and foot"""
    return html.Div(
        children=[
            html.Div(
                children=[dcc.Markdown("**Pulse-Oximetry:** Time-Series Forecast")],
                className="main-header",
            ),
            html.Div(
                children=[
                    dcc.Dropdown(
                        id="dropdown-forecast",
                        options=[
                            {"label": "SPO2 Histogram", "value": "SPO2H"},
                            {"label": "SPO2 Prophet", "value": "SPO2"},
                            {"label": "SPO2 Component", "value": "SPO2C"},
                            {"label": "BPM Histogram", "value": "BPMH"},
                            {"label": "BPM Prophet", "value": "BPM"},
                            {"label": "BPM Component", "value": "BPMC"},
                        ],
                        value="SPO2H",
                        placeholder="Choose New Forecast",
                    ),
                    dcc.Dropdown(
                        id="dropdown-time-window",
                        options=[
                            {"label": "5 min", "value": "5"},
                            {"label": "30 min", "value": "30"},
                            {"label": "1 hr", "value": "60"},
                            {"label": "3 hr", "value": "180"},
                            {"label": "6 hr", "value": "360"},
                            {"label": "12 hr", "value": "720"},
                            {"label": "24 hr", "value": "1440"},
                        ],
                        value="60",
                        placeholder="Choose Duration of Viewing",
                        persistence=True,
                    ),
                    dcc.Dropdown(
                        id="dropdown-agg-window",
                        options=[
                            {"label": "Mean", "value": "mean"},
                            {"label": "Median", "value": "median"},
                            {"label": "Stddev", "value": "stddev"},
                        ],
                        value="mean",
                        placeholder="Choose Aggregation Window",
                        persistence=True,
                    ),
                ],
                className="main-control",
            ),
            html.Div(
                children=[
                    html.Div(dcc.Graph(id="my-plot"), className="card"),
                    html.Div(
                        daq.LEDDisplay(
                            id="spo2-led",
                            label="Current SPO2(%)",
                            labelPosition="bottom",
                            size=64,
                            color="#6fc3df",
                            backgroundColor="#171b26",
                            value="",
                        ),
                        className="card",
                    ),
                    dcc.Loading(
                        id="loading",
                        children=[
                            html.Div(dcc.Graph(id="plot-forecast"), className="card"),
                        ],
                    ),
                    html.Div(
                        daq.LEDDisplay(
                            id="bpm-led",
                            label="Current BPM",
                            labelPosition="bottom",
                            size=64,
                            color="#6fc3df",
                            backgroundColor="#171b26",
                            value="",
                        ),
                        className="card",
                    ),
                ],
                className="main-cards",
            ),
        ],
        className="main",
    )


app.layout = html.Div(
    id="main-app-container",
    children=[
        build_header(),
        build_main(),
        build_footer(),
        # html.Div(id='intermediate-current', style={'display': 'none'}),
        dcc.Interval(id="interval-component", interval=1000 * 5, n_intervals=0,),
        dcc.Interval(id="led-interval-component", interval=1000, n_intervals=0,),
    ],
    className="grid-container",
)

client.__del__()


@app.callback(
    Output("my-plot", "figure"),
    [
        Input("interval-component", "n_intervals"),
        Input("dropdown-time-window", "value"),
        Input("dropdown-agg-window", "value"),
    ],
)
def update_current_plot(dummy, time_value, agg_value):
    period = round(60 / (360 / int(time_value)))
    time_value = round(int(time_value))
    if agg_value == "mean":
        agg_value = "mean"
    elif agg_value == "median":
        agg_value = '(tables=<-, column) => tables |> median(method: "exact_selector")'
    elif agg_value == "stddev":
        agg_value = "stddev"

    df_current_sats = query_api.query_data_frame(
        'from(bucket: "pulse_oximeter") '
        "|> range(start: -" + str(time_value) + "m) "
        '|> filter(fn: (r) => r._measurement == "spo2") '
        "|> aggregateWindow(every: "
        + str(period)
        + "s, fn: "
        + agg_value
        + ", createEmpty: false)"
        '|> drop(columns: ["_start","_stop","_measurement"])'
        '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
    )
    client.__del__()

    try:
        df_current_sats["_time"] = pd.to_datetime(df_current_sats["_time"])
        df_current_sats["_time"] = df_current_sats["_time"].dt.tz_convert("Europe/Oslo")

        fig = create_current_sats_plot(df_current_sats)
    except KeyError:
        fig = go.Figure()
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "DISCONNECTED",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 28},
                }
            ],
        )

    return fig


@app.callback(
    Output("plot-forecast", "figure"), [Input("dropdown-forecast", "value")],
)
def update_forecast(value):
    fig = make_forecast(value)

    return fig


@app.callback(
    [Output("spo2-led", "value"), Output("bpm-led", "value")],
    [Input("led-interval-component", "n_intervals")],
)
def update_leds(dummy):
    df_led = query_api.query_data_frame(
        'from(bucket: "pulse_oximeter")'
        "|> range(start: -10s)"
        '|> filter(fn: (r) => r._measurement == "spo2")'
        '|> drop(columns: ["_start", "_measurement", "_time"])'
        "|> last()"
        '|> pivot(rowKey:["_stop"], columnKey: ["_field"], valueColumn: "_value")'
    )

    client.__del__()

    try:
        return (df_led["spo2"], df_led["bpm"])
    except KeyError:
        return (-1, -1)


@app.callback(
    Output("download-csv", "data"),
    [Input("export-csv-button", "n_clicks")],
    [State("dropdown-time-window", "value")],
    prevent_initial_call=True,
)
def download_csv(dummy, x_disp):
    df_download_csv = query_api.query_data_frame(
        'from(bucket: "pulse_oximeter") '
        "|> range(start: -" + x_disp + ") "
        "|> filter(fn: (r) => "
        '  r._measurement == "spo2") '
        "|> window(every: 1m) "
        "|> mean()"
    )
    client.__del__()

    update_df = sanitize_dataframe(df_download_csv)

    return send_data_frame(update_df.to_csv, "pulse_oximeter_data.csv")


@app.callback(
    Output("download-xlsx", "data"),
    [Input("export-xlsx-button", "n_clicks")],
    [State("dropdown-time-window", "value")],
    prevent_initial_call=True,
)
def download_xlsx(dummy, x_disp):
    df_download_xlsx = query_api.query_data_frame(
        'from(bucket: "pulse_oximeter") '
        "|> range(start: -" + x_disp + ") "
        "|> filter(fn: (r) => "
        '  r._measurement == "spo2") '
        "|> window(every: 1m) "
        "|> mean()"
    )
    client.__del__()

    update_df = sanitize_dataframe(df_download_xlsx)

    return send_data_frame(update_df.to_excel, "pulse_oximeter_data.xlsx")


if __name__ == "__main__":
    app.run_server(debug=True)
