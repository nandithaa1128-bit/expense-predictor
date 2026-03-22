import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

USERS_FILE = "users.csv"

CATEGORIES = [
    "🍔 Food",
    "🚗 Transport",
    "🎮 Entertainment",
    "🛍️ Shopping",
    "💡 Bills",
    "🏥 Health",
    "📦 Other"
]

def init_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password"])


def signup(username, password):
    df = pd.read_csv(USERS_FILE)
    if username in df["username"].values:
        return False
    df.loc[len(df)] = [username, password]
    df.to_csv(USERS_FILE, index=False)
    return True


def login(username, password):
    df = pd.read_csv(USERS_FILE)
    user = df[(df["username"] == username) & (df["password"] == password)]
    return not user.empty



def get_user_file(username):
    return f"{username}_expenses.csv"


def initialize_file(filename):
    if not os.path.exists(filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Date", "Category", "Amount", "Description"])


def load_df(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame(columns=["ID", "Date", "Category", "Amount", "Description"])


def get_budget_file(username):
    return f"{username}_budget.csv"


def save_budget(username, month, amount):
    file = get_budget_file(username)

    if os.path.exists(file):
        df = pd.read_csv(file)
        if "Month" not in df.columns:
            df = pd.DataFrame(columns=["Month", "Budget"])
    else:
        df = pd.DataFrame(columns=["Month", "Budget"])

    df = df[df["Month"] != month]
    df.loc[len(df)] = [month, float(amount)]
    df.to_csv(file, index=False)


def load_budget(username):
    file = get_budget_file(username)

    if os.path.exists(file):
        df = pd.read_csv(file)
        if "Month" not in df.columns:
            return pd.DataFrame(columns=["Month", "Budget"])
        return df

    return pd.DataFrame(columns=["Month", "Budget"])




def detect_anomalies(df):
    """Isolation Forest: marks each row True if anomalous."""
    if len(df) < 5:
        df = df.copy()
        df["Anomaly"] = False
        return df
    le = LabelEncoder()
    df_ml = df.copy()
    df_ml["CategoryEncoded"] = le.fit_transform(df_ml["Category"].astype(str))
    model = IsolationForest(contamination=0.15, random_state=42)
    preds = model.fit_predict(df_ml[["Amount", "CategoryEncoded"]])
    df = df.copy()
    df["Anomaly"] = preds == -1
    return df


def predict_next_month(df):
    """Linear Regression on monthly totals to predict next month."""
    tmp = df.copy()
    tmp["Month"] = tmp["Date"].astype(str).str[:7]
    monthly = tmp.groupby("Month")["Amount"].sum().reset_index().sort_values("Month").reset_index(drop=True)
    if len(monthly) < 2:
        return None
    monthly["Idx"] = np.arange(len(monthly))
    model = LinearRegression()
    model.fit(monthly[["Idx"]], monthly["Amount"])
    return round(max(float(model.predict([[len(monthly)]])[0]), 0), 2)


def forecast_months(df, n=3):
    """Linear Regression: forecast next N months of spending."""
    tmp = df.copy()
    tmp["Month"] = tmp["Date"].astype(str).str[:7]
    monthly = tmp.groupby("Month")["Amount"].sum().reset_index().sort_values("Month").reset_index(drop=True)
    if len(monthly) < 2:
        return []
    monthly["Idx"] = np.arange(len(monthly))
    model = LinearRegression()
    model.fit(monthly[["Idx"]], monthly["Amount"])
    preds = model.predict([[len(monthly) + i] for i in range(n)])
    return [round(max(float(p), 0), 2) for p in preds]


def category_risk(df):
    """Statistical outlier: categories with spend > mean + 1 std."""
    totals = df.groupby("Category")["Amount"].sum()
    mean, std = totals.mean(), totals.std()
    return {cat: total for cat, total in totals.items() if total > mean + std}


def suggest_category(description):
    """Keyword classifier to auto-suggest a category from description."""
    desc = description.lower()
    rules = {
        "🍔 Food": ["food", "restaurant", "eat", "lunch", "dinner", "breakfast", "cafe", "coffee", "pizza", "burger", "swiggy", "zomato"],
        "🚗 Transport": ["uber", "ola", "bus", "train", "fuel", "petrol", "cab", "auto", "metro", "transport", "rapido"],
        "🎮 Entertainment": ["movie", "game", "netflix", "spotify", "concert", "show", "stream", "prime"],
        "🛍️ Shopping": ["amazon", "shop", "buy", "cloth", "shirt", "shoes", "mall", "order", "flipkart", "myntra"],
        "💡 Bills": ["bill", "electricity", "water", "rent", "wifi", "internet", "recharge", "mobile", "broadband"],
        "🏥 Health": ["doctor", "medicine", "hospital", "pharmacy", "gym", "health", "clinic", "tablet"],
    }
    for cat, keywords in rules.items():
        if any(k in desc for k in keywords):
            return cat
    return "📦 Other"

def run_app(username):
    filename = get_user_file(username)
    initialize_file(filename)

    df = load_df(filename)

    st.sidebar.title(f" {username}")

    if st.sidebar.button(" Logout"):
        del st.session_state["user"]
        st.rerun()

    menu = st.sidebar.radio("", [" Add", " Dashboard", "Analytics", " Budget"])

    
    if menu == " Add":
        st.header(" Add Expenses")

        st.caption("Type description + amount → press **Enter** or **Add** → saved instantly to the right category ")

        with st.form("quick_form", clear_on_submit=True):
            col_d, col_desc, col_amt = st.columns([1, 2, 1])
            with col_d:
                q_date = st.date_input("Date", datetime.today())
            with col_desc:
                q_desc = st.text_input("Description", placeholder="e.g. zomato, uber, netflix...")
            with col_amt:
                q_amt = st.text_input("Amount", placeholder="0.00")

            submitted = st.form_submit_button("Add", use_container_width=True)

            if submitted:
                if q_desc and q_amt:
                    try:
                        amt_val = float(q_amt)
                        auto_cat = suggest_category(q_desc)
                        new_id = 1 if df.empty else int(df["ID"].max()) + 1
                        df = pd.concat([df, pd.DataFrame([{
                            "ID": new_id,
                            "Date": str(q_date),
                            "Category": auto_cat,
                            "Amount": amt_val,
                            "Description": q_desc
                        }])], ignore_index=True)
                        df.to_csv(filename, index=False)
                        st.success(f" Added to **{auto_cat}** — ${amt_val:.2f}  |  \"{q_desc}\"")
                    except ValueError:
                        st.warning(" Invalid amount, enter a number.")
                else:
                    st.warning("Fill in both description and amount.")

        df_fresh = load_df(filename)
        if not df_fresh.empty:
            st.caption(" Recent entries:")
            st.dataframe(df_fresh.tail(5)[["Date", "Category", "Amount", "Description"]],
                         use_container_width=True, hide_index=True)

        st.divider()

        with st.expander(" Manual Add — pick categories yourself"):
            with st.form("manual_form", clear_on_submit=True):
                date = st.date_input("Date", datetime.today(), key="man_date")
                expense_data = {}

                cols = st.columns(3)
                for i, cat in enumerate(CATEGORIES):
                    with cols[i % 3]:
                        val = st.text_input(cat, key=f"man_{i}")
                        if val:
                            try:
                                expense_data[cat] = float(val)
                            except:
                                st.warning(f"Invalid value for {cat}")

                desc = st.text_input("Description", key="man_desc")
                submitted_manual = st.form_submit_button("Save", use_container_width=True)

                if submitted_manual:
                    if expense_data:
                        for cat, amt in expense_data.items():
                            new_id = 1 if df.empty else int(df["ID"].max()) + 1
                            df = pd.concat([df, pd.DataFrame([{
                                "ID": new_id,
                                "Date": str(date),
                                "Category": cat,
                                "Amount": amt,
                                "Description": desc
                            }])], ignore_index=True)
                        df.to_csv(filename, index=False)
                        st.success(" Saved!")
                    else:
                        st.warning("Enter at least one value.")

    elif menu == "Dashboard":

        st.header("Dashboard")

        if not df.empty:
            total = df["Amount"].sum()

            # ML: prediction shown as a second metric card next to total
            prediction = predict_next_month(df)
            col_t, col_p = st.columns(2)
            with col_t:
                st.metric("Total Spending", f"${total:.2f}")
            with col_p:
                if prediction:
                    st.metric(" Predicted Next Month", f"${prediction:.2f}",
                              help="Linear Regression trained on your monthly totals")

            df["Month"] = df["Date"].astype(str).str[:7]

            col1, col2 = st.columns(2)

            with col1:
                selected_month = st.selectbox("Month", ["All"] + sorted(df["Month"].unique()))

            with col2:
                selected_category = st.selectbox("Category", ["All"] + CATEGORIES)

            filtered_df = df.copy()

            if selected_month != "All":
                filtered_df = filtered_df[filtered_df["Month"] == selected_month]

            if selected_category != "All":
                filtered_df = filtered_df[filtered_df["Category"] == selected_category]

            if len(filtered_df) >= 5:
                filtered_df = detect_anomalies(filtered_df).reset_index(drop=True)
                anomaly_count = int(filtered_df["Anomaly"].sum())
                anomaly_flags = filtered_df["Anomaly"].tolist()
                display_df = filtered_df.drop(columns=["Anomaly"]).reset_index(drop=True)

                def row_style(row):
                    flag = anomaly_flags[row.name] if row.name < len(anomaly_flags) else False
                    return ["background-color: #ffe0e0; color: #900" if flag else "" for _ in row]

                st.dataframe(
                    display_df.style.apply(row_style, axis=1),
                    use_container_width=True
                )
                if anomaly_count:
                    st.warning(f" {anomaly_count} unusual transaction(s) highlighted in red — detected by **Isolation Forest**.")
            else:
                st.dataframe(filtered_df, use_container_width=True)

            st.write(f"Total: ${filtered_df['Amount'].sum():.2f}")

        else:
            st.warning("No data")

   
    elif menu == "Analytics":

        st.header(" Analytics")

        if not df.empty:

            st.subheader(" Category Distribution")
            cat_data = df.groupby("Category")["Amount"].sum()

            fig, ax = plt.subplots()
            ax.pie(cat_data, labels=cat_data.index, autopct='%1.1f%%')
            st.pyplot(fig)

            
            risky = category_risk(df)
            if risky:
                st.warning(
                    "**High-spend categories** *(spend > mean + 1 std)*: " +
                    ", ".join([f"{cat} (${amt:.2f})" for cat, amt in risky.items()])
                )

            st.subheader(" Monthly Report")
            df["Month"] = df["Date"].astype(str).str[:7]
            monthly = df.groupby("Month")["Amount"].sum()

            st.bar_chart(monthly)

            forecast = forecast_months(df, n=3)
            if forecast:
                st.caption(" **ML Forecast — next 3 months** *(Linear Regression)*")
                forecast_df = pd.DataFrame({
                    "Month": [f"Month +{i+1}" for i in range(3)],
                    "Predicted ($)": forecast
                }).set_index("Month")
                st.bar_chart(forecast_df)

        else:
            st.warning("No data")

    
    elif menu == "Budget":

        st.header("Budget Planner")

        selected_date = st.date_input("Select Month")
        selected_month = selected_date.strftime("%Y-%m")

        budget_input = st.number_input("Set Budget", min_value=0.0)

        if st.button("Save Budget"):
            save_budget(username, selected_month, budget_input)
            st.success(" Budget saved for month!")

        if not df.empty:

            df["Month"] = df["Date"].astype(str).str[:7]
            budget_df = load_budget(username)

            if selected_month in df["Month"].values:

                spent = df[df["Month"] == selected_month]["Amount"].sum()

                budget_row = budget_df[budget_df["Month"] == selected_month]

                if not budget_row.empty:
                    budget = float(budget_row["Budget"].values[0])

                    st.write(f"Month: {selected_month}")
                    st.write(f"Spent: ${spent:.2f}")
                    st.write(f"Budget: ${budget:.2f}")

                    savings = budget - spent

                    if savings > 0:
                        st.success(f"You saved: ${savings:.2f}")
                    else:
                        st.error(f"Overspent by: ${abs(savings):.2f}")

                        month_df = df[df["Month"] == selected_month]
                        cat_total = month_df.groupby("Category")["Amount"].sum()

                        top_cat = cat_total.idxmax()
                        st.warning(f"Highest spending: {top_cat}")

                   
                    prediction = predict_next_month(df)
                    if prediction:
                        st.divider()
                        st.write(" **ML Budget Forecast** *(Linear Regression)*")
                        st.write(f"Predicted spend next month → **${prediction:.2f}**")
                        if prediction > budget:
                            st.error(f" You may **exceed your budget** by **${prediction - budget:.2f}** next month.")
                        else:
                            st.success(f" On track — projected **${budget - prediction:.2f} under budget** next month.")

                else:
                    st.info("No budget set for this month")

            else:
                st.info("No expenses for this month")



def main():
    st.set_page_config(page_title="Expense Tracker", layout="centered")

    init_users()

    if "user" not in st.session_state:

        st.title("Expense Tracker")

        menu = st.radio("Select", ["Login", "Signup"])

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if menu == "Signup":
            if st.button("Create Account"):
                if signup(username, password):
                    st.success("Account created!")
                else:
                    st.error("User already exists")

        else:
            if st.button("Login"):
                if login(username, password):
                    st.session_state["user"] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    else:
        run_app(st.session_state["user"])


if __name__ == "__main__":
    main()