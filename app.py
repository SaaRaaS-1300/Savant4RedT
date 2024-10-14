import streamlit as st


def sidebar_logo():
    st.sidebar.image("resource/logo_2.png", use_column_width=True)


def side_board_A():
    # 侧边栏
    board_1 = st.Page(
        "reports/introduction.py",
        title="Introduction",
        icon="🖼️"
    )

    board_2 = st.Page(
        "reports/guideline.py",
        title="Guideline",
        icon="📝"
    )

    board_3 = st.Page(
        "reports/evaluation.py",
        title="Evaluation",
        icon="⌛"
    )

    return board_1, board_2, board_3


def side_board_B():
    # 侧边栏
    board_1 = st.Page(
        "demonstration/det4txt.py",
        title="Detector4Text",
        icon="🧃"
    )

    board_2 = st.Page(
        "demonstration/det4img_internvl2.py",
        title="Detector4Img by InternVL2",
        icon="🍂"
    )

    return board_1, board_2


if __name__ == "__main__":
    # setting
    st.set_page_config(  # 设置页面配置
        page_title="Homepage",  # 设置网页标题
        page_icon="🌻"
    )
    with open("resource/styles.css", encoding="UTF-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # sidebar + introduction
    sidebar_logo()
    introduction, guideline, evaluation = side_board_A()
    det4txt, det4img = side_board_B()
    st.navigation(
        {
            "Reports": [introduction, guideline, evaluation],
            "Demonstration": [det4txt, det4img]
        }
    ).run()
