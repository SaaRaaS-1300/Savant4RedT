import streamlit as st

# content
st.title("🌠 Savant4RedT ⌈内容安全⌋ Expert")
st.subheader('Overview of the Savant4RedT')

st.markdown(
    "```plaintext\n" +
    "“考量防御成本是信息安全领域的重要问题” -- Defender in the Cyber World\n" +
    "```",
    unsafe_allow_html=True
)
st.markdown("---", unsafe_allow_html=True)
st.image("resource/intro_pic/intro_pic_1.png", use_column_width=True)
st.markdown("---", unsafe_allow_html=True)
st.markdown("## Abstract", unsafe_allow_html=True)
st.markdown("大语言模型 (LLM) 是自然语言处理领域 (NLP) 不可或缺的重要组成部分。随着模型参数的不断扩展和"
            "预训练语料库的不断增加，LLM 在各类 NLP 任务中展示了卓越的能力。但是，在 LLM 不断深入语言领"
            "域深水区域的过程中，各类内容安全问题也逐渐成为工程人员的关注热点。例如，如图 3-1 所示，尽管 "
            "LLM 拥有优秀的指令跟随能力，攻击者依旧可以根据特殊的注入方法破解 LLM 的防御机制；LLM 有时也"
            "可能违背人类的价值观和偏好，输出不真实、有毒、有偏见甚至非法的内容。", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)


# Two columns with images
col1, col2 = st.columns(2)

with col1:
    st.image("resource/intro_pic/col_hacker.gif", caption="Col_Hacker", width=1024, use_column_width=True)
    st.markdown("### Threats of Cyber Attacks")
    st.markdown("大规模语言模型正面临日益复杂的非法攻击威胁。这些攻击包括数据中毒、模型窃取、对抗性攻击等，"
                "旨在破坏模型的正常运行或利用其漏洞获取敏感信息。攻击者可能通过恶意输入、操控训练数据或绕过"
                "安全防护机制，影响模型的决策和输出，造成经济和安全隐患。为应对这些威胁，需要采取多层次的防"
                "御措施，为此我们提出了 Savant4RedT 模型专家组，致力于保障大模型的稳定性和安全性")

with col2:
    st.image("resource/intro_pic/col_coder.gif", caption="Col_Coder", width=1024, use_column_width=True)
    st.markdown("### Hijacking of Evil-Content")
    st.markdown("恶意内容的劫持是指攻击者通过操控输入或利用模型漏洞，迫使大模型生成或传播有害内容的行为。这"
                "类攻击可能利用模型在内容生成时的开放性特点，注入恶意请求，如仇恨言论、虚假信息、恶意软件代"
                "码等。通过劫持模型生成的输出，攻击者不仅可以散布不当信息，还可能导致大模型失去可信度或被用"
                "于非法目的。为防范此类攻击，需加强输入过滤、输出审查以及模型的安全设计，确保其在处理不良请"
                "求时具备识别和阻断能力，从而减少恶意劫持的风险。")

st.markdown("---", unsafe_allow_html=True)
st.markdown("## System Model", unsafe_allow_html=True)
st.markdown("我们将目标需求划分为“检测成本定位”、“检测精度划分”、“数据解构”三个部分进行处理。考虑到我们使用"
            "的硬件设备环境复杂，即实现“通过较低成本代价完成 LLM 专家垂域任务”，我们将场景设定为“轻量级专家"
            "检测”流程", unsafe_allow_html=True)
st.image("resource/intro_pic/intro_pic_2.png", caption="Column 2 Image", use_column_width=True)
st.markdown("该流程分为两个方法，即“用户输入和提取”方法和“系统推理及反馈”方法。其中，“用户输入和提取”可以引"
            "导用户将信息输入至专家系统中，通过收集非结构化数据和引入信息提取规则 (具体见代码仓库) 来实现对"
            "于非结构化信息的提取工作。完成这部分后，“系统推理和反馈”方法会将提取后的查询信息输入至 LLM 中，"
            "获得标准的文档分析结果。其中，文档分析结果使用带有规律性的回答方式 (具体内容见技术特点“特殊数据"
            "结构”)，既可以实现安全对齐，又可以摘取隐含的结构化数据，从而实现信息验证和自动化信息监测。",
            unsafe_allow_html=True)

st.markdown("---")
# GitHub link at the bottom
st.image("resource/logo_1.png", use_column_width=True)
if st.button('🐬 Read for More Details in Github'):
    st.markdown(
        '<meta http-equiv="refresh" content="0; url=https://github.com/SaaRaaS-1300/Savant4RedT">',
        unsafe_allow_html=True
    )
