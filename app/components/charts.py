import plotly.graph_objects as go
import plotly.express as px

PRIMARY_BLUE = "#008CBB"
ACCENT_GOLD = "#fcb004"
SECONDARY_BLUE = "#80d2d4"

CHARCOAL_TEXT = "#0F172A"

SOFT_GREY = "#DCE6F0"
GRID_GREY = "#EEF2F7"

ALERT_RED = "#EF4444"
SUCCESS_GREEN = "#10B981"


def apply_corporate_layout(fig):

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font_family="Outfit, sans-serif",
                      font_color=CHARCOAL_TEXT,
                      title_font=dict(size=16,
                                      color=CHARCOAL_TEXT,
                                      family="Outfit"),
                      margin=dict(l=30, r=30, t=50, b=30),
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1,
                                  font=dict(size=11)))

    fig.update_xaxes(showgrid=True,
                     gridcolor=GRID_GREY,
                     linecolor=SOFT_GREY,
                     tickfont=dict(size=11),
                     title_font=dict(size=12, color=CHARCOAL_TEXT))

    fig.update_yaxes(showgrid=True,
                     gridcolor=GRID_GREY,
                     linecolor=SOFT_GREY,
                     tickfont=dict(size=11),
                     title_font=dict(size=12, color=CHARCOAL_TEXT))

    fig.update_annotations(font_color=CHARCOAL_TEXT)

    return fig


def create_bar_chart(df,
                     x_col,
                     y_col,
                     title="",
                     orientation="v",
                     color_by=None):

    if color_by:

        color_seq = [PRIMARY_BLUE, ACCENT_GOLD, SECONDARY_BLUE]

    else:

        color_seq = [PRIMARY_BLUE]

    fig = px.bar(df,
                 x=x_col,
                 y=y_col,
                 title=title,
                 orientation=orientation,
                 color=color_by,
                 color_discrete_sequence=color_seq)

    fig.update_traces(marker=dict(line=dict(width=0)))

    return apply_corporate_layout(fig)


def create_line_chart(df, x_col, y_col, title="", color_by=None):

    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=color_by,
        color_discrete_sequence=[PRIMARY_BLUE, ACCENT_GOLD, SECONDARY_BLUE])

    fig.update_traces(line=dict(width=3))

    return apply_corporate_layout(fig)


def create_histogram(df, x_col, title="", nbins=30):

    fig = px.histogram(df,
                       x=x_col,
                       title=title,
                       nbins=nbins,
                       color_discrete_sequence=[PRIMARY_BLUE])

    fig.update_traces(marker=dict(line=dict(width=0.5, color="#FFFFFF")))

    return apply_corporate_layout(fig)
