import streamlit as st
from pages import step_0, step_1, step_2, step_3, step_4, step_5, step_6

def main():
    # JavaScript로 스크롤 제어
    st.markdown('<div id="top-of-page"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <script>
            console.log("JavaScript is running!");
            document.getElementById('top-of-page').scrollIntoView({ behavior: 'smooth' });
        </script>
        """,
        unsafe_allow_html=True
    )

    # app.py에서 초기화된 st.session_state.step 사용
    step = st.session_state.step

    if step == 0:
        step_0.app()
    elif step == 1:
        step_1.app()
    elif step == 2:
        step_2.app()
    elif step == 3:
        step_3.app()
    elif step == 4:
        step_4.app()
    elif step == 5:
        step_5.app()
    elif step == 6:
        step_6.app()

if __name__ == "__main__":
    main()