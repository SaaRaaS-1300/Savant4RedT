import cpuinfo
import psutil
import pynvml
import streamlit as st


def sidebar_logo():
    st.sidebar.image("resource/logo_2.png", use_column_width=True)
    monitor_cpu_gpu()


def monitor_cpu_gpu():
    # init
    pynvml.nvmlInit()
    gpu_devices = pynvml.nvmlDeviceGetCount()
    gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_devices)]
    gpu_names = [pynvml.nvmlDeviceGetName(handle) for handle in gpu_handles]
    # gpu
    utilization = [pynvml.nvmlDeviceGetUtilizationRates(handle).gpu for handle in gpu_handles]
    memory_used = [pynvml.nvmlDeviceGetMemoryInfo(handle).used for handle in gpu_handles]
    memory_total = [pynvml.nvmlDeviceGetMemoryInfo(handle).total for handle in gpu_handles]
    memory_used = [f"{round(memory / 1024 ** 3, 2)} GB" for memory in memory_used]
    memory_total = [f"{round(memory / 1024 ** 3, 2)} GB" for memory in memory_total]
    # cpu
    cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
    cpu_utilization = psutil.cpu_percent(interval=1)
    cpu_memory = psutil.virtual_memory()
    cpu_memory_used = round(cpu_memory.used / 1024 ** 3, 2)
    cpu_memory_total = round(cpu_memory.total / 1024 ** 3, 2)
    # render
    st.sidebar.markdown("### System Monitor")
    info = '| Devices | Utilization | Memory (Used) |\n| --- | --- | --- |\n'
    info += f"| {cpu_name} | {cpu_utilization}% | {cpu_memory_used}/{cpu_memory_total} GB |\n"
    for i in range(gpu_devices):
        info += f"| {gpu_names[i]} | {utilization[i]}% | {memory_used[i]}/{memory_total[i]} GB |\n"
    st.sidebar.markdown(info)


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
        "demonstration/v1_gpu_det4txt.py",
        title="Detector4Text",
        icon="🧃"
    )

    board_2 = st.Page(
        "demonstration/v1_internvl2_det4img.py",
        title="Detector4Img by InternVL2",
        icon="🍂"
    )

    board_3 = st.Page(
        "demonstration/v2_cpu_judge_rewrite.py",
        title="Judge & Rewrite",
        icon="🧃"
    )
    return board_1, board_2, board_3


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
    v1_gpu_det4txt, v1_internvl2_det4img, v2_cpu_judge_rewrite = side_board_B()
    st.navigation(
        {
            "Reports": [introduction, guideline, evaluation],
            "Demonstration": [v1_gpu_det4txt, v1_internvl2_det4img, v2_cpu_judge_rewrite]
        }
    ).run()
