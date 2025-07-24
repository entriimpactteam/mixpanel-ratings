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
    if isinstance(text, str):
        return emoji.replace_emoji(text, replace='')
    return text

def clean_data(df):
    # Handle NaN and non-numeric values in numeric columns
    for col in df.columns[3:]:
        # Convert to string, remove '%' and invalid entries
        df[col] = df[col].astype(str).str.replace('%', '', regex=False).replace('#DIV/0!', pd.NA)
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Replace NaN with 0 for calculations
        df[col] = df[col].fillna(0)
    return df

def clean_display_name(name):
    """Remove 'Total' from vertical or category names for display."""
    return str(name).replace('Total', '').strip() if pd.notna(name) else ''

def process_and_generate(vertical_df, category_df, course_df):
    try:
        # Extract headers
        vertical_headers_last = vertical_df.iloc[1, 0:15].tolist()
        vertical_headers_this = vertical_df.iloc[1, 16:31].tolist()
        category_headers_last = category_df.iloc[1, 0:15].tolist()
        category_headers_this = category_df.iloc[1, 16:31].tolist()
        course_headers_this = course_df.iloc[1, 16:31].tolist()
        course_headers_last = course_df.iloc[1, 0:15].tolist()

        # Process dataframes
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

        # Rename columns
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

        # Forward fill to handle NaN in identifier columns
        for df in [vertical_data_last, vertical_data_this, category_data_last, category_data_this, course_data_this, course_data_last]:
            df.iloc[:, 0] = df.iloc[:, 0].ffill()
            df.iloc[:, 1] = df.iloc[:, 1].ffill()
            if df.shape[1] > 2:
                df.iloc[:, 2] = df.iloc[:, 2].ffill()

        # Clean numeric data
        vertical_data_last = clean_data(vertical_data_last)
        vertical_data_this = clean_data(vertical_data_this)
        category_data_last = clean_data(category_data_last)
        category_data_this = clean_data(category_data_this)
        course_data_this = clean_data(course_data_this)
        course_data_last = clean_data(course_data_last)

        # Debugging: Check dataframe shapes
        st.write("Dataframe Shapes:")
        st.write(f"vertical_data_this: {vertical_data_this.shape}, vertical_data_last: {vertical_data_last.shape}")
        st.write(f"category_data_this: {category_data_this.shape}, category_data_last: {category_data_last.shape}")
        st.write(f"course_data_this: {course_data_this.shape}, course_data_last: {course_data_last.shape}")

        def is_no_ratings(val):
            return pd.isna(val) or val == 0

        def flag_courses(df_this, df_last):
            flagged = []
            for _, row in df_this.iterrows():
                flags = []
                course_name = row['Course']
                row_last = df_last[df_last['Course'] == course_name]
                row_last = row_last.iloc[0] if not row_last.empty else pd.Series(dtype=float)

                # Helper function to process each rating type
                def process_rating(rating_col, count_col, flag_symbol, flag_label):
                    rating = row[rating_col]
                    if not is_no_ratings(rating) and rating < 4.5:
                        count = int(row[count_col]) if not is_no_ratings(row[count_col]) else 0
                        is_persistent = not row_last.empty and not is_no_ratings(row_last.get(rating_col, pd.NA)) and row_last[rating_col] < 4.5
                        flag_text = (
                            f'<span style="color:red;" title="Persistent issue">{flag_symbol} {flag_label}: {rating:.2f} ({count})</span>'
                            if is_persistent else
                            f'{flag_symbol} {flag_label}: {rating:.2f} ({count})'
                        )
                        flags.append(flag_text)

                # Process each rating type
                process_rating('Weighted Average of Live sessions', 'SUM of No: of live ratings', '🔵', 'Live')
                process_rating('Weighted Average of Mentor Ratings', 'SUM of No: of mentor ratings', '🟣', 'Mentor')
                process_rating('Weighted Average of Course Ratings', 'SUM of No: of Course Ratings', '🟠', 'Course')
                process_rating('Weighted Average of VOD Ratings', 'SUM of No: of VOD Ratings', '🟢', 'VOD')
                process_rating('Weighted Average of Live Record ratings', 'SUM of No: Of live record ratings', '🔵', 'Live Rec')

                # Sessions Rated
                sessions_rated = row['AVERAGE of % of rated live sessions']
                if not is_no_ratings(sessions_rated) and sessions_rated < 80:
                    flags.append(f"⚠️ Sessions Rated: {sessions_rated:.2f}%")

                # Sessions <3.5
                sessions_below_3_5 = row['AVERAGE of % of live sessions rated below 3.5']
                if not is_no_ratings(sessions_below_3_5) and sessions_below_3_5 > 0:
                    flags.append(f"🚩 Sessions with <3.5 Rating: {sessions_below_3_5:.2f}%")

                if flags:
                    flagged.append({
                        'Vertical': row['Vertical'],
                        'Category': row['Category'],
                        'Course': row['Course'],
                        'Flags': ' | '.join(flags)
                    })
            return pd.DataFrame(flagged)

        flagged_courses = flag_courses(course_data_this, course_data_last)

        # Debugging: Display categories and flagged courses
        st.write("Categories in category_data_this:")
        st.write(category_data_this[['Vertical', 'Category']].to_string())
        st.write("Flagged Courses:")
        st.write(flagged_courses[['Vertical', 'Category', 'Course', 'Flags']].to_string())

        def normalize_vertical_name(name):
            return str(name).replace('Total', '').strip().lower() if pd.notna(name) else ''

        def normalize_category_name(name):
            return str(name).replace('Total', '').strip().lower() if pd.notna(name) else ''

        vertical_data_this['Vertical_norm'] = vertical_data_this['Vertical'].apply(normalize_vertical_name)
        vertical_data_last['Vertical_norm'] = vertical_data_last['Vertical'].apply(normalize_vertical_name)
        category_data_this['Vertical_norm'] = category_data_this['Vertical'].apply(normalize_vertical_name)
        category_data_last['Vertical_norm'] = category_data_last['Vertical'].apply(normalize_vertical_name)
        flagged_courses['Vertical_norm'] = flagged_courses['Vertical'].apply(normalize_vertical_name)
        category_data_this['Category_norm'] = category_data_this['Category'].apply(normalize_category_name)
        category_data_last['Category_norm'] = category_data_last['Category'].apply(normalize_category_name)
        flagged_courses['Category_norm'] = flagged_courses['Category'].apply(normalize_category_name)
        course_data_this['Category_norm'] = course_data_this['Category'].apply(normalize_category_name)

        def plot_bar(df_this, df_last, label_col, title, filename):
            if df_this.empty or df_last.empty:
                st.warning(f"Skipping {title} chart due to empty dataframe.")
                return None
            df_this = df_this.copy()
            df_last = df_last.copy()

            # Calculate total ratings, handling NaN
            rating_cols = [
                'SUM of No: of live ratings', 'SUM of No: of mentor ratings',
                'SUM of No: of Course Ratings', 'SUM of No: of VOD Ratings',
                'SUM of No: Of live record ratings'
            ]
            df_this['Total Ratings'] = df_this[rating_cols].sum(axis=1, skipna=True).fillna(0)
            df_last['Total Ratings'] = df_last[rating_cols].sum(axis=1, skipna=True).fillna(0)

            changes = []
            labels = []
            for idx, row in df_this.iterrows():
                if label_col == 'Category' and 'Vertical' in df_this.columns:
                    match = df_last[(df_last['Vertical'] == row['Vertical']) & (df_last['Category'] == row['Category'])]
                else:
                    match = df_last[df_last[label_col] == row[label_col]]
                last = match['Total Ratings'].iloc[0] if not match.empty else 0
                change = row['Total Ratings'] - last
                changes.append(change)
                label = f"{clean_display_name(row[label_col])} ({int(row['Total Ratings'])})" if pd.notna(row[label_col]) else f"Unknown ({int(row['Total Ratings'])})"
                labels.append(label)

            # Sort data
            sorted_data = sorted(zip(labels, changes), key=lambda x: x[1])
            labels, changes = zip(*sorted_data)
            colors = ['green' if c >= 0 else 'red' for c in changes]

            # Create plot
            fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.7)))
            bars = ax.barh(labels, changes, color=colors, edgecolor='black', height=0.6)
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel('Change in Number of Ratings', fontsize=12)
            ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
            ax.grid(axis='x', linestyle='--', alpha=0.7)

            # Add text labels
            for bar, change in zip(bars, changes):
                xpos = bar.get_width() + (3 if change >= 0 else -3)
                align = 'left' if change >= 0 else 'right'
                ax.text(xpos, bar.get_y() + bar.get_height()/2, f'{int(change)}', va='center', ha=align, fontsize=12, fontweight='bold')

            green_patch = mpatches.Patch(color='green', label='Green: Growth')
            red_patch = mpatches.Patch(color='red', label='Red: Decline')
            ax.legend(handles=[green_patch, red_patch], title="Legend", loc='lower right')
            plt.tight_layout(rect=[0, 0, 1, 1])

            # Save plot
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
            rating_cols = [
                'SUM of No: of live ratings', 'SUM of No: of mentor ratings',
                'SUM of No: of Course Ratings', 'SUM of No: of VOD Ratings',
                'SUM of No: Of live record ratings'
            ]

            # Group by Vertical to ensure all categories are covered
            grouped = vertical_data_this.groupby('Vertical')
            for v_idx, (vertical, v_group) in enumerate(grouped):
                vertical_norm = normalize_vertical_name(vertical)
                vertical_display = clean_display_name(vertical)
                # Calculate total ratings for vertical from vertical_data_this
                total_this = v_group[rating_cols].sum().sum(skipna=True)
                total_this = int(total_this) if pd.notna(total_this) else 0

                # Calculate total ratings for vertical from vertical_data_last
                v_last = vertical_data_last[vertical_data_last['Vertical_norm'] == vertical_norm]
                total_last = v_last[rating_cols].sum().sum(skipna=True)
                total_last = int(total_last) if pd.notna(total_last) else 0

                # Get sessions rated and sessions below 3.5
                sessions_rated_this = v_group['AVERAGE of % of rated live sessions'].iloc[0] if not v_group.empty and 'AVERAGE of % of rated live sessions' in v_group.columns else 0
                sessions_below_3_5_this = v_group['AVERAGE of % of live sessions rated below 3.5'].iloc[0] if not v_group.empty and 'AVERAGE of % of live sessions rated below 3.5' in v_group.columns else 0
                sessions_rated_last = v_last['AVERAGE of % of rated live sessions'].iloc[0] if not v_last.empty and 'AVERAGE of % of rated live sessions' in v_last.columns else 0
                sessions_below_3_5_last = v_last['AVERAGE of % of live sessions rated below 3.5'].iloc[0] if not v_last.empty and 'AVERAGE of % of live sessions rated below 3.5' in v_last.columns else 0

                change = total_this - total_last
                direction = "Increased by" if change > 0 else "Decreased by" if change < 0 else "No Change"
                output.append(f'<h2>Vertical {v_idx+1} - <b>{vertical_display}</b></h2>')
                output.append(
                    f'<p>{direction} <b>{abs(change)}</b> ratings (<b>{total_this}</b> this week). '
                    f'Sessions Rated: <b>{sessions_rated_this:.2f}%</b> ({sessions_rated_last:.2f}% last week). '
                    f'Sessions with <3.5 Rating: <b>{sessions_below_3_5_this:.2f}%</b> ({sessions_below_3_5_last:.2f}% last week).</p>'
                )

                # Categories under this vertical
                v_categories = category_data_this[category_data_this['Vertical_norm'] == vertical_norm]
                output.append('<h3>Categories</h3>')
                for _, c_row in v_categories.iterrows():
                    category = c_row['Category']
                    category_norm = c_row['Category_norm']
                    category_display = clean_display_name(category)
                    category_total_this = c_row[rating_cols].sum(skipna=True)
                    category_total_this = int(category_total_this) if pd.notna(category_total_this) else 0

                    c_row_last = category_data_last[(category_data_last['Vertical_norm'] == vertical_norm) &
                                                   (category_data_last['Category_norm'] == category_norm)]
                    if not c_row_last.empty:
                        category_total_last = c_row_last[rating_cols].sum().sum(skipna=True)
                        category_total_last = int(category_total_last) if pd.notna(category_total_last) else 0
                    else:
                        category_total_last = 0

                    cat_change = category_total_this - category_total_last
                    cat_direction = "increased by" if cat_change > 0 else "decreased by" if cat_change < 0 else "remained unchanged with"
                    cat_change_text = f"{abs(cat_change)} ratings" if cat_change != 0 else "no change in ratings"
                    cat_rated = c_row['AVERAGE of % of rated live sessions']
                    cat_rated = cat_rated if pd.notna(cat_rated) else 0
                    cat_below_3_5 = c_row['AVERAGE of % of live sessions rated below 3.5']
                    cat_below_3_5 = cat_below_3_5 if pd.notna(cat_below_3_5) else 0

                    flagged = flagged_courses[(flagged_courses['Vertical_norm'] == vertical_norm) &
                                             (flagged_courses['Category_norm'] == category_norm)]
                    flagged_html = ''
                    if not flagged.empty:
                        table = ['<table border="1" cellpadding="0" cellspacing="0"><tr><th>Flagged Courses</th><th>Flags</th></tr>']
                        for _, f_row in flagged.iterrows():
                            flag_cells = ''.join(f'<div style="display:inline-block;padding:2px 4px;">{flag}</div>'
                                                 for flag in f_row["Flags"].split(' | ') if flag)
                            table.append(f'<tr><td style="white-space:nowrap;"><b>{f_row["Course"]}</b></td><td style="white-space:nowrap;padding:0 2px;">{flag_cells}</td></tr>')
                        table.append('</table>')
                        flagged_html = ''.join(table)

                    # Debugging: Log category details
                    st.write(f"Processing Category: {vertical_display} - {category_display}")
                    st.write(f"Flagged Courses: {len(flagged)} found")
                    st.write(f"Category Total This Week: {category_total_this}, Last Week: {category_total_last}, Change: {cat_change}")
                    st.write(f"Sessions Rated: {cat_rated:.2f}%, Sessions <3.5: {cat_below_3_5:.2f}%")

                    # Detailed commentary for each category
                    commentary = []
                    commentary.append(f"The category <b>{category_display}</b> {cat_direction} {cat_change_text}, with a total of <b>{category_total_this}</b> ratings this week.")
                    if cat_rated > 0:
                        commentary.append(f"{cat_rated:.2f}% of live sessions were rated, indicating {'strong' if cat_rated >= 80 else 'moderate' if cat_rated >= 50 else 'low'} engagement.")
                    else:
                        commentary.append("No live sessions were rated this week.")
                    if cat_below_3_5 > 0:
                        commentary.append(f"{cat_below_3_5:.2f}% of live sessions received ratings below 3.5, suggesting areas for improvement.")
                    else:
                        commentary.append("No live sessions were rated below 3.5, reflecting positive performance.")

                    cat_block = (
                        f'<h4><b>{category_display}</b></h4>'
                        f'<p>{" ".join(commentary)}</p>'
                        f'<p>⚠️ <b>Sessions Rated:</b> {cat_rated:.2f}% 🚩 <b>Sessions with <3.5 Rating:</b> {cat_below_3_5:.2f}%</p>'
                        + (flagged_html if flagged_html else '<p>No flagged courses.</p>')
                    )
                    output.append(cat_block)

            return '\n'.join(output)

        html_summary_output = make_html_summary()

        def image_to_base64(image_path):
            if not image_path or not os.path.exists(image_path):
                st.warning(f"Image {image_path} not found, using placeholder.")
                return ""
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
            'h4 { font-size: 1.0em; margin-top: 10px; }'
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
            st.success("Report generated successfully! Click the link to download the HTML file. Open it in a browser and use Ctrl+P or Cmd+P to save as PDF.")
            os.unlink(tmp_file_path)
            if verticals_img:
                os.unlink(verticals_img)
            if categories_img:
                os.unlink(categories_img)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please check the uploaded files and ensure they match the expected format (e.g., Sheet1.csv for vertical data, Sheet2.csv for categories, Sheet3.csv for courses).")

if file_vertical and file_category and file_course:
    vertical_df = pd.read_csv(file_vertical, header=None)
    category_df = pd.read_csv(file_category, header=None)
    course_df = pd.read_csv(file_course, header=None)

    process_and_generate(vertical_df, category_df, course_df)
else:
    st.info("Please upload all three CSV files to generate the reports.")
