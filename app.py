import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Data Analysis System", layout="wide")
st.title("Sales Data Analysis & Prediction System")
st.caption("Interactive sales analytics dashboard with forecasting")

# Colours used across all charts so the same category or region always looks the same
PALETTE = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#0891b2", "#db2777"]
GRID = "rgba(128,128,128,0.25)"


@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    else:
        data = pd.read_csv(Path("data/sales_data.csv"))
    required = ["Order_ID", "Date", "Product", "Category", "Region", "Quantity", "Sales", "Profit"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for c in ["Quantity", "Sales", "Profit"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["Date", "Sales", "Profit", "Quantity"]).copy()
    data = data.drop_duplicates()
    data = data[data["Quantity"] > 0]
    data["Month"] = data["Date"].dt.to_period("M").astype(str)
    return data


def inr_short(n):
    """Indian-style compact currency: 1.30 Cr, 4.5 L, 12,345."""
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1e7:
        return f"{sign}₹{n/1e7:.2f} Cr"
    if n >= 1e5:
        return f"{sign}₹{n/1e5:.2f} L"
    return f"{sign}₹{n:,.0f}"


def nice_ticks(vmax, n=5):
    """Round, readable tick values from 0 up to vmax."""
    if vmax <= 0:
        return [0]
    raw = vmax / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return list(np.arange(0, vmax + step, step))


def money_axis(fig, vmax, axis="y", title=None):
    """Show axis values as Rs / L / Cr instead of long raw numbers."""
    ticks = nice_ticks(vmax)
    kwargs = dict(tickvals=ticks, ticktext=[inr_short(t) for t in ticks], range=[0, ticks[-1]])
    if title:
        kwargs["title_text"] = title
    if axis == "y":
        fig.update_yaxes(**kwargs)
    else:
        fig.update_xaxes(**kwargs)


def style(fig, title, x_title=None, y_title=None, legend=True, height=420):
    """Common look for every chart: clear title, axis labels, gridlines, legend below."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=18), x=0, xanchor="left"),
        font=dict(size=13),
        height=height,
        margin=dict(l=10, r=20, t=60, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_size=13),
        colorway=PALETTE,
    )
    fig.update_xaxes(title_text=x_title, showgrid=False, linecolor=GRID, automargin=True)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def money_table(df, money_cols, pct_cols=(), int_cols=()):
    fmt = {c: "₹{:,.0f}" for c in money_cols}
    fmt.update({c: "{:.1f}%" for c in pct_cols})
    fmt.update({c: "{:,.0f}" for c in int_cols})
    st.dataframe(df.style.format(fmt), width="stretch", hide_index=True)


with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload sales CSV", type=["csv"])
    try:
        data = load_data(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()
    min_d, max_d = data["Date"].min().date(), data["Date"].max().date()
    date_range = st.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    if len(date_range) != 2:
        st.stop()
    cats = st.multiselect("Category", sorted(data["Category"].unique()), default=sorted(data["Category"].unique()))
    regs = st.multiselect("Region", sorted(data["Region"].unique()), default=sorted(data["Region"].unique()))

f = data[(data["Date"].dt.date >= date_range[0]) & (data["Date"].dt.date <= date_range[1]) &
         data["Category"].isin(cats) & data["Region"].isin(regs)].copy()

sales = f["Sales"].sum()
profit = f["Profit"].sum()
orders = f["Order_ID"].nunique()
qty = f["Quantity"].sum()
aov = sales / orders if orders else 0
margin = profit / sales * 100 if sales else 0

# ---------- KPI cards ----------
KPI_CSS = """
<style>
.kpi-card{position:relative;overflow:hidden;border-radius:12px;padding:18px 18px 14px 18px;
  background:rgba(128,128,128,0.07);border:1px solid rgba(128,128,128,0.22);min-height:128px;}
.kpi-card::before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--accent);}
.kpi-label{font-size:0.8rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;opacity:.75;margin-bottom:8px;}
.kpi-value{font-size:1.9rem;font-weight:700;line-height:1.15;margin-bottom:6px;white-space:nowrap;}
.kpi-sub{font-size:0.85rem;opacity:.75;line-height:1.35;}
.kpi-bar{height:6px;border-radius:99px;background:rgba(128,128,128,0.25);margin-top:10px;overflow:hidden;}
.kpi-bar>span{display:block;height:100%;border-radius:99px;background:var(--accent);}
</style>
"""
st.markdown(KPI_CSS, unsafe_allow_html=True)


def kpi_card(label, value, sub, accent, full_value=None, bar_pct=None):
    bar = ""
    if bar_pct is not None:
        bar = f'<div class="kpi-bar"><span style="width:{max(0, min(bar_pct, 100)):.1f}%"></span></div>'
    title = f' title="{full_value}"' if full_value else ""
    return (
        f'<div class="kpi-card" style="--accent:{accent};"{title}>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>{bar}</div>'
    )


if f.empty:
    st.warning("No data for the selected filters. Change the date range, category or region.")
    st.stop()

profit_per_unit = profit / qty if qty else 0
units_per_order = qty / orders if orders else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(kpi_card("Total Sales", inr_short(sales), f"Avg order value {inr_short(aov)}",
                     PALETTE[0], full_value=f"₹{sales:,.2f}"), unsafe_allow_html=True)
c2.markdown(kpi_card("Total Profit", inr_short(profit), f"{inr_short(profit_per_unit)} profit per unit",
                     PALETTE[1], full_value=f"₹{profit:,.2f}"), unsafe_allow_html=True)
c3.markdown(kpi_card("Orders", f"{orders:,}", f"{units_per_order:.1f} units per order",
                     PALETTE[3]), unsafe_allow_html=True)
c4.markdown(kpi_card("Units Sold", f"{qty:,.0f}", f"{inr_short(sales/qty if qty else 0)} avg price per unit",
                     PALETTE[2]), unsafe_allow_html=True)
c5.markdown(kpi_card("Profit Margin", f"{margin:.1f}%", f"{inr_short(profit)} profit on {inr_short(sales)} sales",
                     PALETTE[4], bar_pct=margin), unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Products", "Regions", "Customers", "Forecast"])

# Same colour for the same category in every chart
cat_colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(data["Category"].unique()))}
reg_colors = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(sorted(data["Region"].unique()))}

with tab1:
    monthly = f.groupby("Month", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    monthly["MonthDate"] = pd.to_datetime(monthly["Month"] + "-01")
    fig = px.line(monthly, x="MonthDate", y="Sales", markers=True)
    fig.update_traces(line=dict(width=3, color=PALETTE[0]), marker=dict(size=8),
                      hovertemplate="%{x|%B %Y}<br>Sales: ₹%{y:,.0f}<extra></extra>")
    style(fig, "Monthly Sales Trend", "Month", "Sales", legend=False)
    fig.update_xaxes(tickformat="%b %Y", tickangle=-45)
    money_axis(fig, monthly["Sales"].max() * 1.1, "y")
    show(fig)

    a, b = st.columns(2)
    with a:
        cat = f.groupby("Category", as_index=False).agg(Sales=("Sales", "sum")).sort_values("Sales", ascending=False)
        fig = px.bar(cat, x="Category", y="Sales", color="Category", color_discrete_map=cat_colors,
                     text=cat["Sales"].map(inr_short))
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}<br>Sales: ₹%{y:,.0f}<extra></extra>")
        style(fig, "Sales by Category", "Category", "Sales", legend=False)
        money_axis(fig, cat["Sales"].max() * 1.15, "y")
        show(fig)
    with b:
        fig = px.scatter(f, x="Sales", y="Profit", size="Quantity", size_max=14, color="Category",
                         color_discrete_map=cat_colors, opacity=0.6,
                         hover_data={"Product": True, "Region": True, "Quantity": True,
                                     "Sales": ":,.0f", "Profit": ":,.0f"})
        style(fig, "Sales vs Profit per Order", "Sales per order", "Profit per order")
        money_axis(fig, f["Sales"].max() * 1.05, "x")
        ymax = f["Profit"].max() * 1.05
        yt = nice_ticks(ymax)
        fig.update_yaxes(tickvals=yt, ticktext=[inr_short(t) for t in yt])
        show(fig)

with tab2:
    prod = f.groupby(["Product", "Category"], as_index=False).agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum"))
    prod["Profit Margin %"] = np.where(prod["Sales"] != 0, prod["Profit"] / prod["Sales"] * 100, 0)
    top = prod.nlargest(10, "Sales")
    fig = px.bar(top, x="Sales", y="Product", orientation="h", color="Category",
                 color_discrete_map=cat_colors, text=top["Sales"].map(inr_short))
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}<br>Sales: ₹%{x:,.0f}<extra></extra>")
    style(fig, "Top 10 Products by Sales", "Sales", None, height=460)
    fig.update_yaxes(categoryorder="total ascending", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    money_axis(fig, top["Sales"].max() * 1.2, "x")
    show(fig)
    st.markdown("**Product details**")
    money_table(prod.sort_values("Sales", ascending=False), ["Sales", "Profit"],
                pct_cols=["Profit Margin %"], int_cols=["Quantity"])

with tab3:
    reg = f.groupby("Region", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                                                  Orders=("Order_ID", "nunique"))
    a, b = st.columns(2)
    with a:
        fig = px.pie(reg, names="Region", values="Sales", hole=0.45, color="Region",
                     color_discrete_map=reg_colors)
        fig.update_traces(textinfo="label+percent", textposition="outside", sort=False,
                          hovertemplate="%{label}<br>Sales: ₹%{value:,.0f} (%{percent})<extra></extra>")
        style(fig, "Regional Sales Share", legend=False)
        show(fig)
    with b:
        reg_sorted = reg.sort_values("Profit", ascending=False)
        fig = px.bar(reg_sorted, x="Region", y="Profit", color="Region", color_discrete_map=reg_colors,
                     text=reg_sorted["Profit"].map(inr_short))
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}<br>Profit: ₹%{y:,.0f}<extra></extra>")
        style(fig, "Profit by Region", "Region", "Profit", legend=False)
        money_axis(fig, reg_sorted["Profit"].max() * 1.15, "y")
        show(fig)
    st.markdown("**Region details**")
    money_table(reg.sort_values("Sales", ascending=False), ["Sales", "Profit"], int_cols=["Orders"])

with tab4:
    st.info("The sample dataset contains order-level sales rather than personal customer identifiers. "
            "This tab therefore focuses on order-value analysis.")
    daily = f.groupby("Date", as_index=False).agg(Sales=("Sales", "sum"), Orders=("Order_ID", "nunique"))
    daily["Average Order Value"] = daily["Sales"] / daily["Orders"]
    daily["30-day average"] = daily["Average Order Value"].rolling(30, min_periods=7).mean()
    fig = go.Figure()
    fig.add_scatter(x=daily["Date"], y=daily["Average Order Value"], mode="lines", name="Daily",
                    line=dict(width=1, color="rgba(37,99,235,0.35)"),
                    hovertemplate="%{x|%d %b %Y}<br>Daily: ₹%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=daily["Date"], y=daily["30-day average"], mode="lines", name="30-day average",
                    line=dict(width=3, color=PALETTE[2]),
                    hovertemplate="%{x|%d %b %Y}<br>30-day avg: ₹%{y:,.0f}<extra></extra>")
    style(fig, "Average Order Value Over Time", "Date", "Average order value")
    fig.update_xaxes(tickformat="%b %Y")
    money_axis(fig, daily["Average Order Value"].max() * 1.05, "y")
    show(fig)
    st.metric("Average Order Value", f"₹{aov:,.0f}")

with tab5:
    st.subheader("Next 6-Month Sales Forecast")
    monthly_all = f.groupby("Month", as_index=False)["Sales"].sum()
    monthly_all["MonthDate"] = pd.to_datetime(monthly_all["Month"] + "-01")
    monthly_all = monthly_all.sort_values("MonthDate")
    if len(monthly_all) < 4:
        st.warning("Need at least 4 months of data for forecasting.")
    else:
        X = np.arange(len(monthly_all)).reshape(-1, 1)
        y = monthly_all["Sales"].values
        model = LinearRegression().fit(X, y)
        pred_train = model.predict(X)
        mae = mean_absolute_error(y, pred_train)
        rmse = mean_squared_error(y, pred_train) ** 0.5
        future_n = 6
        future_x = np.arange(len(monthly_all), len(monthly_all) + future_n).reshape(-1, 1)
        future_pred = np.maximum(0, model.predict(future_x))
        future_dates = pd.date_range(monthly_all["MonthDate"].iloc[-1] + pd.offsets.MonthBegin(1),
                                     periods=future_n, freq="MS")
        forecast = pd.DataFrame({"Month": future_dates.strftime("%b %Y"), "Forecast Sales": future_pred})

        fig = go.Figure()
        fig.add_scatter(x=monthly_all["MonthDate"], y=monthly_all["Sales"], mode="lines+markers",
                        name="Actual sales", line=dict(width=3, color=PALETTE[0]), marker=dict(size=8),
                        hovertemplate="%{x|%B %Y}<br>Actual: ₹%{y:,.0f}<extra></extra>")
        # start the forecast line at the last actual point so the two lines connect
        fx = [monthly_all["MonthDate"].iloc[-1]] + list(future_dates)
        fy = [monthly_all["Sales"].iloc[-1]] + list(future_pred)
        fig.add_scatter(x=fx, y=fy, mode="lines+markers", name="Forecast",
                        line=dict(width=3, color=PALETTE[2], dash="dash"), marker=dict(size=8),
                        hovertemplate="%{x|%B %Y}<br>Forecast: ₹%{y:,.0f}<extra></extra>")
        style(fig, "Historical and Forecast Sales", "Month", "Sales")
        fig.update_xaxes(tickformat="%b %Y", tickangle=-45)
        money_axis(fig, max(y.max(), future_pred.max()) * 1.1, "y")
        show(fig)

        p1, p2, p3 = st.columns(3)
        p1.metric("Next Month Forecast", f"₹{future_pred[0]:,.0f}")
        p2.metric("Model MAE", f"₹{mae:,.0f}")
        p3.metric("Model RMSE", f"₹{rmse:,.0f}")
        st.markdown("**Forecast table**")
        money_table(forecast, ["Forecast Sales"])

st.divider()
st.subheader("Export Filtered Data")
st.download_button("Download CSV", f.to_csv(index=False).encode("utf-8"), "filtered_sales_data.csv", "text/csv")
st.caption("Educational project. The forecast is a baseline model and should not be treated as a business guarantee.")
