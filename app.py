import streamlit as st
import pandas as pd
import emoji
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import tempfile
import base64

st.set_page_config(page_title="Weekly Ratings Report Generator", layout="wide")

st.title("Weekly Ratings Report Generator")
st.write("Upload your 3 CSV files to generate the reports.")

# --- File uploaders ---
file_vertical = st.file_uploader("Upload Sheet1.csv (Vertical)", type="csv")
file_category = st.file_uploader("Upload Sheet2.csv (Category)", type="csv")
file_course = st.file_uploader("Upload Sheet3.csv (Course)", type="csv")

# Set pandas future option to avoid downcasting warning
pd.set_option('future.no_silent_downcasting', True)

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
    course_headers_last = course_df.iloc[1, 0:15].tolist()

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
    course_data_last = course_df.drop([0, 1]).reset_index(drop=True).iloc[:, 0:15]
    course_data_last.columns = course_headers_last

    course_data_this.rename(columns={
        course_data_this.columns[0]: 'Vertical',
        course_data_this.columns[1]: 'Category',
        course_data_this.columns[2]: 'Course'
    }, inplace=True)
    course_data_last.rename(columns={
        course_data_last.columns[0]: 'Vertical',
        course_data_last.columns[1]: 'Category',
        course_data_last.columns[2]: 'Course'
    }, inplace=True)

    for df in [vertical_data_last, vertical_data_this, category_data_last, category_data_this, course_data_this, course_data_last]:
        df.iloc[:, 0] = df.iloc[:, 0].ffill()
        df.iloc[:, 1] = df.iloc[:, 1].ffill()
        df.iloc[:, 2] = df.iloc[:, 2].ffill()

    vertical_data_last = clean_data(vertical_data_last)
    vertical_data_this = clean_data(vertical_data_this)
    category_data_last = clean_data(category_data_last)
    category_data_this = clean_data(category_data_this)
    course_data_this = clean_data(course_data_this)
    course_data_last = clean_data(course_data_last)

    def is_no_ratings(val):
        return pd.isna(val)

    def flag_courses(df_this, df_last):
        flagged = []
        for _, row in df_this.iterrows():
            flags = []
            course_name = row['Course']
            row_last = df_last[df_last['Course'] == course_name]
            if not row_last.empty:
                row_last = row_last.iloc[0]
            else:
                row_last = pd.Series()

            # Live
            if not is_no_ratings(row['Weighted Average of Live sessions']) and row['Weighted Average of Live sessions'] < 4.5:
                count = int(row['SUM of No: of live ratings']) if not is_no_ratings(row['SUM of No: of live ratings']) else 0
                is_persistent = not row_last.empty and not is_no_ratings(row_last.get('Weighted Average of Live sessions')) and row_last['Weighted Average of Live sessions'] < 4.5
                flag_text = f'<span style="color:red;" title="Persistent issue">🔵 Live: {row["Weighted Average of Live sessions"]:.2f} ({count})</span>' if is_persistent else f'🔵 Live: {row["Weighted Average of Live sessions"]:.2f} ({count})'
                flags.append(flag_text)
            # Mentor
            if not is_no_ratings(row['Weighted Average of Mentor Ratings']) and row['Weighted Average of Mentor Ratings'] < 4.5:
                count = int(row['SUM of No: of mentor ratings']) if not is_no_ratings(row['SUM of No: of mentor ratings']) else 0
                is_persistent = not row_last.empty and not is_no_ratings(row_last.get('Weighted Average of Mentor Ratings')) and row_last['Weighted Average of Mentor Ratings'] < 4.5
                flag_text = f'<span style="color:red;" title="Persistent issue">🟣 Mentor: {row["Weighted Average of Mentor Ratings"]:.2f} ({count})</span>' if is_persistent else f'🟣 Mentor: {row["Weighted Average of Mentor Ratings"]:.2f} ({count})'
                flags.append(flag_text)
            # Course
            if not is_no_ratings(row['Weighted Average of Course Ratings']) and row['Weighted Average of Course Ratings'] < 4.5:
                count = int(row['SUM of No: of Course Ratings']) if not is_no_ratings(row['SUM of No: of Course Ratings']) else 0
                is_persistent = not row_last.empty and not is_no_ratings(row_last.get('Weighted Average of Course Ratings')) and row_last['Weighted Average of Course Ratings'] < 4.5
                flag_text = f'<span style="color:red;" title="Persistent issue">🟠 Course: {row["Weighted Average of Course Ratings"]:.2f} ({count})</span>' if is_persistent else f'🟠 Course: {row["Weighted Average of Course Ratings"]:.2f} ({count})'
                flags.append(flag_text)
            # VOD
            if not is_no_ratings(row['Weighted Average of VOD Ratings']) and row['Weighted Average of VOD Ratings'] < 4.5:
                count = int(row['SUM of No: of VOD Ratings']) if not is_no_ratings(row['SUM of No: of VOD Ratings']) else 0
                is_persistent = not row_last.empty and not is_no_ratings(row_last.get('Weighted Average of VOD Ratings')) and row_last['Weighted Average of VOD Ratings'] < 4.5
                flag_text = f'<span style="color:red;" title="Persistent issue">🟢 VOD: {row["Weighted Average of VOD Ratings"]:.2f} ({count})</span>' if is_persistent else f'🟢 VOD: {row["Weighted Average of VOD Ratings"]:.2f} ({count})'
                flags.append(flag_text)
            # Live Record
            if not is_no_ratings(row['Weighted Average of Live Record ratings']) and row['Weighted Average of Live Record ratings'] < 4.5:
                count = int(row['SUM of No: Of live record ratings']) if not is_no_ratings(row['SUM of No: Of live record ratings']) else 0
                is_persistent = not row_last.empty and not is_no_ratings(row_last.get('Weighted Average of Live Record ratings')) and row_last['Weighted Average of Live Record ratings'] < 4.5
                flag_text = f'<span style="color:red;" title="Persistent issue">🔵 Live Rec: {row["Weighted Average of Live Record ratings"]:.2f} ({count})</span>' if is_persistent else f'🔵 Live Rec: {row["Weighted Average of Live Record ratings"]:.2f} ({count})'
                flags.append(flag_text)
            # Sessions Rated
            if not is_no_ratings(row['AVERAGE of % of rated live sessions']) and row['AVERAGE of % of rated live sessions'] < 80:
                flags.append(f"⚠️ Sessions Rated: {row['AVERAGE of % of rated live sessions']:.2f}%")
            # Sessions <3.5
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

    flagged_courses = flag_courses(course_data_this, course_data_last)

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

    def plot_bar(df_this, df_last, label_col, title, filename):
        if df_this.empty:
            return
        df_this = df_this.copy()
        df_last = df_last.copy()
        # Calculate Total Ratings using only rating count columns, summing non-NaN values
        df_this['Total Ratings'] = df_this[['SUM of No: of live ratings', 'SUM of No: of mentor ratings', 
                                           'SUM of No: of Course Ratings', 'SUM of No: of VOD Ratings', 
                                           'SUM of No: Of live record ratings']].sum(axis=1, skipna=True)
        df_last['Total Ratings'] = df_last[['SUM of No: of live ratings', 'SUM of No: of mentor ratings', 
                                           'SUM of No: of Course Ratings', 'SUM of No: of VOD Ratings', 
                                           'SUM of No: Of live record ratings']].sum(axis=1, skipna=True)
        if label_col == 'Category' and 'Vertical' in df_this.columns:
            changes = []
            labels = []
            for idx, row in df_this.iterrows():
                match = df_last[(df_last['Vertical'] == row['Vertical']) & (df_last['Category'] == row['Category'])]
                last = match['Total Ratings'].values[0] if not match.empty else 0
                change = row['Total Ratings'] - last
                changes.append(change)
                # Safely handle NaN (from #DIV/0! in weighted averages) by using 0 for display
                total_ratings = int(row['Total Ratings']) if not pd.isna(row['Total Ratings']) else 0
                labels.append(f"{row['Category']} ({total_ratings})")
        else:
            changes = []
            labels = []
            for idx, row in df_this.iterrows():
                match = df_last[df_last[label_col] == row[label_col]]
                last = match['Total Ratings'].values[0] if not match.empty else 0
                change = row['Total Ratings'] - last
                changes.append(change)
                total_ratings = int(row['Total Ratings']) if not pd.isna(row['Total Ratings']) else 0
                labels.append(f"{row[label_col]} ({total_ratings})")
        colors = ['green' if c >= 0 else 'red' for c in changes]
        sorted_data = sorted(zip(labels, changes, colors), key=lambda x: x[1])
        labels, changes, colors = zip(*sorted_data)
        fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.7)))
        bars = ax.barh(labels, changes, color=colors, edgecolor='black', height=0.6)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('Change in Number of Ratings', fontsize=12)
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        for bar, change in zip(bars, changes):
            xpos = bar.get_width() + (3 if change >= 0 else -3)
            align = 'left' if change >= 0 else 'right'
            ax.text(xpos, bar.get_y() + bar.get_height()/2, f'{int(change)}', va='center', ha=align, fontsize=12, fontweight='bold')
        green_patch = mpatches.Patch(color='green', label='Green: Growth')
        red_patch = mpatches.Patch(color='red', label='Red: Decline')
        ax.legend(handles=[green_patch, red_patch], title="Legend", loc='lower right')
        plt.tight_layout(rect=[0, 0, 1, 1])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            plt.savefig(tmp_file.name, bbox_inches='tight')
            plt.close()
        return tmp_file.name

    # Generate charts
    verticals_img = plot_bar(
        vertical_data_this, vertical_data_last, 'Vertical',
        "Vertical Performance: Ratings Change and This Week's Total",
        "verticals_img"
    )
    categories_img = plot_bar(
        category_data_this, category_data_last, 'Category',
        "Category Performance: Ratings Change and This Week's Total",
        "categories_img"
    )

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
            direction = "Increased by" if change > 0 else "Decreased by" if change < 0 else "No Change"
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
                cat_direction = "Up" if cat_change > 0 else "Down" if cat_change < 0 else "No Change"
                cat_change_text = f"+{cat_change} ratings" if cat_change > 0 else f"{cat_change} ratings" if cat_change < 0 else "0 ratings"
                cat_rated = c_row['AVERAGE of % of rated live sessions']
                cat_below_3_5 = c_row['AVERAGE of % of live sessions rated below 3.5']
                flagged = flagged_courses[(flagged_courses['Vertical_norm'] == vertical_norm) & (flagged_courses['Category_norm'] == category_norm)]
                flagged_html = ''
                if not flagged.empty:
                    table = ['<table border="1" cellpadding="0" cellspacing="0"><tr><th>Flagged Courses</th><th>Flags</th></tr>']
                    for _, f_row in flagged.iterrows():
                        flag_cells = ''
                        for flag in f_row["Flags"].split(' | '):
                            flag_cells += f'<div style="display:inline-block;padding:2px 4px;">{flag}</div>'
                        table.append(f'<tr><td style="white-space:nowrap;"><b>{f_row["Course"]}</b></td><td style="white-space:nowrap;padding:0 2px;">{flag_cells}</td></tr>')
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

    # Convert images to base64 for HTML
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    verticals_base64 = image_to_base64(verticals_img)
    categories_base64 = image_to_base64(categories_img)

    # Generate HTML
    html_output = (
        '<html><head><meta charset="UTF-8">'
        '<style>'
        'body { font-family: Arial, Helvetica, sans-serif; margin: 20px; }'
        'h2 { font-size: 1.5em; margin-top: 20px; }'
        'h3 { font-size: 1.2em; margin-top: 15px; }'
        'p { margin: 5px 0; }'
        'table, th, td { border: 1px solid black; border-collapse: collapse; padding: 5px; font-size: 1em; white-space: nowrap; }'
        'th { background-color: #f2f2f2; }'
        'img { max-width: 100%; height: auto; }'
        '@media print { body { margin: 0; } img { page-break-inside: avoid; } }'
        '</style></head><body>'
        f'<img src="data:image/png;base64,{verticals_base64}" alt="Vertical Performance" style="max-width:600px;display:block;margin-bottom:10px;"><br>'
        f'<img src="data:image/png;base64,{categories_base64}" alt="Category Performance" style="max-width:600px;display:block;margin-bottom:20px;"><br>'
        + remove_emojis(html_summary_output) +
        '</body></html>'
    )

    # Download button for HTML
    if st.button("Download Report"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as tmp_file:
            tmp_file.write(html_output)
            tmp_file_path = tmp_file.name
        with open(tmp_file_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="weekly_ratings_report.html">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        st.success("Click the link to download the report as an HTML file. Open it in a browser and use Ctrl+P or Cmd+P to save as PDF.")
        os.unlink(tmp_file_path)  # Clean up temporary file
        os.unlink(verticals_img)
        os.unlink(categories_img)

if file_vertical and file_category and file_course:
    vertical_df = pd.read_csv(file_vertical, header=None)
    category_df = pd.read_csv(file_category, header=None)
    course_df = pd.read_csv(file_course, header=None)

    process_and_generate(vertical_df, category_df, course_df)
else:
    st.info("Please upload all three CSV files to generate the reports.")
