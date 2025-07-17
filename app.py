import streamlit as st
import pandas as pd
import emoji
from weasyprint import HTML

st.set_page_config(page_title="Weekly Ratings Report Generator", layout="centered")

st.title("Weekly Ratings Report Generator")
st.write("Upload your 3 CSV files to generate the reports.")

# --- File uploaders ---
file_vertical = st.file_uploader("Upload Sheet1.csv (Vertical)", type="csv")
file_category = st.file_uploader("Upload Sheet2.csv (Category)", type="csv")
file_course = st.file_uploader("Upload Sheet3.csv (Course)", type="csv")

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def clean_data(df):
    for col in df.columns[3:]:
        df[col] = df[col].astype(str).str.replace('%', '')
        df[col] = df[col].replace('#DIV/0!', pd.NA)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def process_and_generate(vertical_df, category_df, course_df):
    vertical_headers_last = vertical_df.iloc[1, 0:15].tolist()
    vertical_headers_this = vertical_df.iloc[1, 16:31].tolist()
    category_headers_last = category_df.iloc[1, 0:15].tolist()
    category_headers_this = category_df.iloc[1, 16:31].tolist()
    course_headers_this = course_df.iloc[1, 16:31].tolist()

    vertical_data_last = vertical_df.drop([0, 1]).reset_index(drop=True).iloc[:, 0:15]
    vertical_data_last.columns = vertical_headers_last
    vertical_data_this = vertical_df.drop([0, 1]).reset_index(drop=True).iloc[:, 16:31]
    vertical_data_this.columns = vertical_headers_this
    category_data_last = category_df.drop([0, 1]).reset_index(drop=True).iloc[:, 0:15]
    category_data_last.columns = category_headers_last
    category_data_this = category_df.drop([0, 1]).reset_index(drop=True).iloc[:, 16:31]
    category_data_this.columns = category_headers_this
    course_data_this = course_df.drop([0, 1]).reset_index(drop=True).iloc[:, 16:31]
    course_data_this.columns = course_headers_this
    course_data_this.rename(columns={
        course_data_this.columns[0]: 'Vertical',
        course_data_this.columns[1]: 'Category',
        course_data_this.columns[2]: 'Course'
    }, inplace=True)

    for df in [vertical_data_last, vertical_data_this, category_data_last, category_data_this, course_data_this]:
        df.iloc[:, 0] = df.iloc[:, 0].ffill()
        df.iloc[:, 1] = df.iloc[:, 1].ffill()
        df.iloc[:, 2] = df.iloc[:, 2].ffill()

    vertical_data_last = clean_data(vertical_data_last)
    vertical_data_this = clean_data(vertical_data_this)
    category_data_last = clean_data(category_data_last)
    category_data_this = clean_data(category_data_this)
    course_data_this = clean_data(course_data_this)

    def is_no_ratings(val):
        return pd.isna(val)

    def flag_courses(df):
        flagged = []
        for _, row in df.iterrows():
            flags = []
            if all(is_no_ratings(row[col]) for col in [
                'Weighted Average of Live sessions',
                'Weighted Average of Mentor Ratings',
                'Weighted Average of Course Ratings',
                'Weighted Average of VOD Ratings',
                'Weighted Average of Live Record ratings',
                'AVERAGE of % of rated live sessions',
                'AVERAGE of % of live sessions rated below 3.5']):
                continue
            if not is_no_ratings(row['Weighted Average of Live sessions']) and row['Weighted Average of Live sessions'] < 4.5:
                flags.append(f"🔵 Live: {row['Weighted Average of Live sessions']:.2f}")
            if not is_no_ratings(row['Weighted Average of Mentor Ratings']) and row['Weighted Average of Mentor Ratings'] < 4.5:
                flags.append(f"🟣 Mentor: {row['Weighted Average of Mentor Ratings']:.2f}")
            if not is_no_ratings(row['Weighted Average of Course Ratings']) and row['Weighted Average of Course Ratings'] < 4.5:
                flags.append(f"🟠 Course: {row['Weighted Average of Course Ratings']:.2f}")
            if not is_no_ratings(row['Weighted Average of VOD Ratings']) and row['Weighted Average of VOD Ratings'] < 4.5:
                flags.append(f"🟢 VOD: {row['Weighted Average of VOD Ratings']:.2f}")
            if not is_no_ratings(row['Weighted Average of Live Record ratings']) and row['Weighted Average of Live Record ratings'] < 4.5:
                flags.append(f"🔵 Live Rec: {row['Weighted Average of Live Record ratings']:.2f}")
            if not is_no_ratings(row['AVERAGE of % of rated live sessions']) and row['AVERAGE of % of rated live sessions'] < 80:
                flags.append(f"⚠️ Sessions Rated: {row['AVERAGE of % of rated live sessions']:.2f}%")
            if not is_no_ratings(row['AVERAGE of % of live sessions rated below 3.5']) and row['AVERAGE of % of live sessions rated below 3.5'] > 0:
                flags.append(f"🚩 Sessions with <3.5 Rating: {row['AVERAGE of % of live sessions rated below 3.5']:.2f}%")
            if flags:
                flagged.append({
                    'Vertical': row['Vertical'],
                    'Category': row['Category'],
                    'Course': row['Course'],
                    'Flags': ' | '.join(flags)
                })
        return pd.DataFrame(flagged)

    flagged_courses = flag_courses(course_data_this)

    def normalize_vertical_name(name):
        return str(name).replace('Total', '').strip().lower()

    vertical_data_this['Vertical_norm'] = vertical_data_this['Vertical'].apply(normalize_vertical_name)
    vertical_data_last['Vertical_norm'] = vertical_data_last['Vertical'].apply(normalize_vertical_name)
    category_data_this['Vertical_norm'] = category_data_this['Vertical'].apply(normalize_vertical_name)
    category_data_last['Vertical_norm'] = category_data_last['Vertical'].apply(normalize_vertical_name)
    flagged_courses['Vertical_norm'] = flagged_courses['Vertical'].apply(normalize_vertical_name)

    def normalize_category_name(name):
        return str(name).replace('Total', '').strip().lower()

    category_data_this['Category_norm'] = category_data_this['Category'].apply(normalize_category_name)
    category_data_last['Category_norm'] = category_data_last['Category'].apply(normalize_category_name)
    flagged_courses['Category_norm'] = flagged_courses['Category'].apply(normalize_category_name)
    course_data_this['Category_norm'] = course_data_this['Category'].apply(normalize_category_name)

    def make_html_summary():
        output = []
        for v_idx, v_row in vertical_data_this.iterrows():
            vertical = v_row['Vertical']
            vertical_norm = v_row['Vertical_norm']
            total_this = int(v_row['SUM of No: of live ratings'] + v_row['SUM of No: of mentor ratings'] + v_row['SUM of No: of Course Ratings'] + v_row['SUM of No: of VOD Ratings'] + v_row['SUM of No: Of live record ratings'])
            v_row_last = vertical_data_last[vertical_data_last['Vertical_norm'] == vertical_norm]
            if not v_row_last.empty:
                v_row_last = v_row_last.iloc[0]
                total_last = int(v_row_last['SUM of No: of live ratings'] + v_row_last['SUM of No: of mentor ratings'] + v_row_last['SUM of No: of Course Ratings'] + v_row_last['SUM of No: of VOD Ratings'] + v_row_last['SUM of No: Of live record ratings'])
            else:
                total_last = 0
            change = total_this - total_last
            direction = "📈 Increased by" if change > 0 else "📉 Decreased by" if change < 0 else "📊 No Change"
            output.append(f'<h2>Vertical {v_idx+1} - <b>{vertical}</b></h2>')
            output.append(f'<p>{direction} <b>{change}</b> ratings (<b>{total_this}</b> this week).</p>')
            v_categories_this = category_data_this[category_data_this['Vertical_norm'] == vertical_norm]
            pos_cats = []
            neg_cats = []
            for _, c_row in v_categories_this.iterrows():
                category = c_row['Category']
                category_norm = c_row['Category_norm']
                category_total_this = int(c_row['SUM of No: of live ratings'] + c_row['SUM of No: of mentor ratings'] + c_row['SUM of No: of Course Ratings'] + c_row['SUM of No: of VOD Ratings'] + c_row['SUM of No: Of live record ratings'])
                c_row_last = category_data_last[(category_data_last['Vertical_norm'] == vertical_norm) & (category_data_last['Category_norm'] == category_norm)]
                if not c_row_last.empty:
                    c_row_last = c_row_last.iloc[0]
                    category_total_last = int(c_row_last['SUM of No: of live ratings'] + c_row_last['SUM of No: of mentor ratings'] + c_row_last['SUM of No: of Course Ratings'] + c_row_last['SUM of No: of VOD Ratings'] + c_row_last['SUM of No: Of live record ratings'])
                else:
                    category_total_last = 0
                cat_change = category_total_this - category_total_last
                cat_direction = "📈" if cat_change > 0 else "📉" if cat_change < 0 else "📊"
                cat_change_text = f"+{cat_change} ratings" if cat_change > 0 else f"{cat_change} ratings" if cat_change < 0 else "0 ratings"
                cat_rated = c_row['AVERAGE of % of rated live sessions']
                cat_below_3_5 = c_row['AVERAGE of % of live sessions rated below 3.5']
                flagged = flagged_courses[(flagged_courses['Vertical_norm'] == vertical_norm) & (flagged_courses['Category_norm'] == category_norm)]
                flagged_html = ''
                if not flagged.empty:
                    table = ['<table border=\"1\" cellpadding=\"0\" cellspacing=\"0\"><tr><th>Flagged Courses</th><th>Flags</th></tr>']
                    for _, f_row in flagged.iterrows():
                        table.append(f'<tr><td style="white-space:nowrap;"><b>{f_row["Course"]}</b></td><td style="white-space:nowrap;padding:0 2px;">{f_row["Flags"]}</td></tr>')
                    table.append('</table>')
                    flagged_html = ''.join(table)
                cat_block = (
                    f'<h3><b>{category}</b>: {cat_direction} {cat_change_text} (<b>{category_total_this}</b> this week)</h3>'
                    f'<p>⚠️ <b>Sessions Rated:</b> {cat_rated:.2f}% 🚩 <b>Sessions with <3.5 Rating:</b> {cat_below_3_5:.2f}%</p>'
                    + flagged_html
                )
                if cat_change >= 0 and flagged_html:
                    pos_cats.append(cat_block)
                elif cat_change < 0 and flagged_html:
                    neg_cats.append(cat_block)
            if pos_cats:
                output.append('<h3>Positive Contributors</h3>')
                output.extend(pos_cats)
            if neg_cats:
                output.append('<h3>Negative Contributors</h3>')
                output.extend(neg_cats)
        return '\n'.join(output)

    html_summary_output = make_html_summary()

    html_with_emojis = (
        '<html><head><meta charset=\"UTF-8\"></head><body>'
        + html_summary_output +
        '</body></html>'
    )
    html_noemojis = (
        '<html><head><meta charset=\"UTF-8\"></head><body>'
        + remove_emojis(html_summary_output) +
        '</body></html>'
    )

    return html_with_emojis, html_noemojis

if file_vertical and file_category and file_course:
    vertical_df = pd.read_csv(file_vertical, header=None)
    category_df = pd.read_csv(file_category, header=None)
    course_df = pd.read_csv(file_course, header=None)

    html_with_emojis, html_noemojis = process_and_generate(vertical_df, category_df, course_df)

    st.subheader("Download Reports")
    st.download_button(
        label="Download HTML (with emojis)",
        data=html_with_emojis,
        file_name="Weekly_Ratings_Summary_Robust.html",
        mime="text/html"
    )
    st.download_button(
        label="Download HTML (no emojis)",
        data=html_noemojis,
        file_name="Weekly_Ratings_Summary_Robust_noemojis.html",
        mime="text/html"
    )

    # --- PDF Generation (no emojis) ---
    pdf_bytes = HTML(string=html_noemojis).write_pdf()
    st.download_button(
        label="Download PDF (no emojis)",
        data=pdf_bytes,
        file_name="Weekly_Ratings_Summary_Robust_noemojis.pdf",
        mime="application/pdf"
    )

    st.success("Reports generated! Download using the buttons above.")

else:
    st.info("Please upload all three CSV files to generate the reports.")