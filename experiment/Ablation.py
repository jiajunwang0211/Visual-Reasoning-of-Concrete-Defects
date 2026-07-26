import os
import json
import base64
import copy
import re
from openai import OpenAI
import csv

# ==========================================
# 1. 配置区域 (请填入你的真实信息)
# ==========================================
API_KEY = "sk-A2xlPMiM7BhUdZCJE1B4A425C4444bA3B35f2e4bF053Ec79"  # 你的 API Key
BASE_URL = "https://api.gpt.ge/v1/"  # 你的 API Base URL (如果是中转代理请修改)
MODEL_NAME = "gpt-4o"  # 你使用的模型名称
IMAGE_DIR = r"C:\Users\admin\Desktop\15\data_final\气孔"  # 替换为你的图片文件夹路径
TARGET_CATEGORY = "气孔"
ONTOLOGY_PATH = r"C:\Users\admin\Desktop\15\第三轮\step3_defect_refined_output\step3_defect_refined_ontology（修改）.json"  # 你的真实本体字典路径
GROUND_TRUTH_FILE = r"C:\Users\admin\Desktop\15\data_final\气孔\Bughole.json"
OUTPUT_CSV_PATH = r"C:\Users\admin\Desktop\小论文文献\实验\图片\气孔\消融实验结果新1.csv"
# 所有的可选病害类别 (让阶段二的主治医师做选择题)
ALL_CATEGORIES = ["定向裂缝", "网状裂缝", "钢筋裸露", "剥落", "起皮", "崩解", "蜂窝", "水渍", "铁锈", "生物生长污渍","泛碱", "结壳", "钟乳石状析出", "气孔", "冷缝"]  # 请补全你的类别
# 初始化大模型客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ------------------------------------------
# 加载并解析带语义定义的本体字典 (包含 S_int 和 S_ctx)
# ------------------------------------------
try:
    with open(ONTOLOGY_PATH, 'r', encoding='utf-8') as f:
        full_ontology = json.load(f)

    # 🌟 核心修改：将 S_int 和 S_ctx 合并成一个用于查询的大字典
    ONTOLOGY_COMBINED = {}
    ONTOLOGY_COMBINED.update(full_ontology.get("S_int", {}))
    ONTOLOGY_COMBINED.update(full_ontology.get("S_ctx", {}))

    print("✅ 带有定义的富结构本体字典 (S_int + S_ctx) 合并加载成功！")
except Exception as e:
    print(f"❌ 读取本地本体字典失败，请检查路径: {e}")
    ONTOLOGY_COMBINED = {}

try:
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        ALL_GROUND_TRUTHS = json.load(f)
    print("✅ 类别总基准数据集加载成功！")
except Exception as e:
    print(f"❌ 读取总基准数据失败，请检查路径: {e}")
    ALL_GROUND_TRUTHS = {}

# 你的真实权重字典 (这里以定向裂缝为例，请根据你的实际设定调整)
WEIGHTS = {
    "定向裂缝": {
        "Global_Shape": 3,
        "Feature_Orientation": 1,
        "Branching_Pattern": 3,
        "Path_Regularity": 1
    },
    "钟乳石状析出": {
        "Surface_Attachments": 3,
        "Surface_Orientation": 3,
        "Surface_Color": 1,
        "Local_Profile": 3
    },
    "网状裂缝": {
        "Global_Shape": 3,
        "Branching_Pattern": 3,
        "Path_Regularity": 1
    },
    "蜂窝": {
        "Material_Exposure": 3,
        "Surface_Texture": 3,
        "Porosity": 3,
        "Local_Profile": 1
    },
    "钢筋裸露": {
        "Material_Exposure": 3,
        "Local_Profile": 1
    },
    "冷缝": {
        "Path_Regularity": 1,
        "Relative_Position": 3,
        "Global_Shape": 3,
        "Local_Profile": 1
    },
    "崩解": {
        "Surface_Attachments": 3,
        "Material_Exposure": 3,
        "Edge_Condition": 3,
        "Local_Profile": 1
    },
    "剥落": {
        "Local_Profile": 3,
        "Material_Exposure": 3,
        "Edge_Condition": 1,
        "Surface_Texture": 1
    },
    "泛碱": {
        "Surface_Color": 3,
        "Surface_Attachments": 3,
        "Edge_Condition": 1
    },
    "水渍": {
        "Surface_Color": 3,
        "Edge_Condition": 3,
        "Stain_Pattern": 3,
        "Global_Shape": 1
    },
    "起皮": {
        "Local_Profile": 3,
        "Material_Exposure": 3,
        "Edge_Condition": 1,
        "Surface_Texture": 1
    },
    "生物生长污渍": {
        "Surface_Color": 3,
        "Surface_Attachments": 3,
        "Stain_Pattern": 1
    },
    "气孔": {
        "Local_Profile": 3,
        "Material_Exposure": 3,
        "Porosity": 1,
        "Edge_Condition": 1
    },
    "结壳": {
        "Local_Profile": 3,
        "Material_Exposure": 3,
        "Edge_Condition": 1
    },
    "铁锈": {
        "Surface_Color": 3,
        "Stain_Pattern": 1
    }
}


# ==========================================
# 2. 辅助函数：编码、解析与铁面无私的评分器
# ==========================================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
#
#
# def extract_json_from_text(text):
#     """防弹版 JSON 解析器"""
#     try:
#         match = re.search(r'\{.*\}', text, re.DOTALL)
#         if match:
#             return json.loads(match.group(0))
#         return {}
#     except:
#         return {}

def extract_json_from_text(text):
    """终极防弹版 JSON 解析器"""
    try:
        # 1. 清理 GPT 喜欢加的 markdown 标记
        cleaned_text = text.replace('```json', '').replace('```', '').strip()
        # 2. 匹配大括号内的所有内容
        match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}
    except Exception as e:
        print(f"JSON 解析失败: {e} \n原始文本为: {text}") # 打印出来看看 GPT 到底说了什么鬼话
        return {}


def calculate_wasc_with_gt(predicted_attrs, ground_truth, weights):
    """
    基于基准测试(Ground Truth)的真实评分器：
    支持 GT 数据集中的“软匹配”（当真实值为 List 列表时，命中其一即得分）
    """
    total_score = 0.0
    max_possible_score = sum([weights.get(k, 0) for k in ground_truth.keys()])

    if max_possible_score == 0:
        return 0.0

    for attr, true_val in ground_truth.items():
        pred_val = predicted_attrs.get(attr)

        # 基础过滤：如果预测值是绝对的空对象或空字符串，不给分
        if pred_val in [None, "", "Unknown"]:
            continue

        # ------------------ 优雅的软匹配逻辑 ------------------
        is_correct = False

        # 1. 如果 GT 里的答案是一个列表（候选池），只要预测值在列表里就算对
        if isinstance(true_val, list):
            if pred_val in true_val:
                is_correct = True

        # 2. 如果 GT 里的答案是一个普通的单一字符串，则严格匹配
        else:
            if pred_val == true_val:
                is_correct = True
        # -----------------------------------------------------

        # 如果判定为正确，加上该属性的权重分
        if is_correct:
            total_score += weights.get(attr, 0)

    return round((total_score / max_possible_score) * 100, 2)


def run_two_stage_experiment():
    print(f"\n🚀 初始化【两阶段解耦+基准校验】消融实验: {TARGET_CATEGORY} ...\n")
    print("=" * 65)

    groups = {
        "Group_0_Baseline": None,
        "Group_1_Mask_W3": 3,
        "Group_2_Mask_W1": 1,
        "Group_3_Mask_W0": 0
    }

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]

    experiment_results = []

    for filename in image_files:
        image_path = os.path.join(IMAGE_DIR, filename)
        base64_image = encode_image(image_path)

        base_name = os.path.splitext(filename)[0]
        ground_truth = ALL_GROUND_TRUTHS.get(base_name)
        if not ground_truth:
            ground_truth = ALL_GROUND_TRUTHS.get(filename)

        if not ground_truth:
            print(f"⚠️ 找不到图片 {filename} 在 JSON 中的对应基准数据，已跳过。")
            continue

        print(f"\n🔬 正在测试图片: {filename}")

        # 合并实际 GT 用于后续判卷
        actual_gt_attrs = {}
        if 'S_int' in ground_truth: actual_gt_attrs.update(ground_truth['S_int'])
        if 'S_ctx' in ground_truth: actual_gt_attrs.update(ground_truth['S_ctx'])
        if not actual_gt_attrs: actual_gt_attrs = ground_truth

        # ========================================================
        # 🌟 步骤 A：基准特征全量提取 (只调用 1 次视觉 API)
        # ========================================================
        print("  -> [步骤 A] 执行全量视觉特征提取 (生成 Baseline)...")

        # 构造全量字典
        current_dict_full = {}
        for attr, details in ONTOLOGY_COMBINED.items():
            current_dict_full[attr] = {
                "定义": details["definition"],
                "可选值": details["visual_anchors"]
            }

        prompt_stage1 = f"""
        你是一个严谨的视觉特征提取器。请观察土木工程病害图片，严格按照以下字典提取物理属性。
        【字典(包含定义和必须遵守的可选值)】: 
        {json.dumps(current_dict_full, ensure_ascii=False, indent=2)}

        要求：
        1. 仔细观察图片，只能在"可选值"的列表里挑选最符合的一个词语作为属性值。
        2. 极其重要：如果图片中根本不存在该属性（例如没有金属、没有渗水痕迹），或者由于图片质量无法判断，请你绝对不要瞎猜！你必须输出字符串 "None"！尽管列表中没有这个选项。
        3. 只输出一个 JSON 格式的结果（键为属性名，值为挑选的词或 "None"）。
        4. 绝对不允许猜测、输出或暗示病害的最终类别！
        5. 视觉防混淆指南：
           - 提取 Relative_Position 时：只有当你清晰看到线状痕迹两侧存在材质、颜色或浇筑批次的明显差异，或具备平直切割特征时，才可选 "Joint"。对于普通的不规则开裂，必须输出 "None"。
           - 颜色观测：如果表面有明显的白色覆盖物或析出物，Surface_Color 必须如实提取为 "White"。当你提取 Surface_Color 时，绝对不允许输出周围正常混凝土/白墙的背景色（如 White, Light Gray）。你必须更加关注发生变色、水晕、沉积的那条边缘或斑块本身的颜色，优先提取
           - 附着物观测：如果是白色的粉末或结晶，提取为 "Crystalline" ；如果是粗糙颗粒，提取为 "Granular"；如果没有明显附着物，提取为 "None"。
           - 轮廓观测：只有当表面存在边界清晰的实体凹陷、厚度差或坑洞时，才可选 "Depression"。对于纯粹的表面颜色变化或污渍，Local_Profile 必须保持 "None"。
           - 如果污渍呈现“中间颜色较浅，四周边缘有一圈颜色更深、更清晰的轮廓线”的蒸发沉积特征，这强烈表明它是夹杂泥沙的水分蒸发留下的。此时：- `Stain_Pattern` 必须优先提取为 "Coffee_Ring"。- `Edge_Condition` 虽然整体可能呈现弥散，但若沉积轮廓明显，亦可提取为 "Sharp"。
           - Local_Profile：请仔细观察破损区域的形态！如果是浅层的、表面砂浆像皮肤一样呈薄片状剥离、脱落，必须输出 Layered_Flake！不要填 None！当你发现表面有起皮 (Layered_Flake) 时，请务必往破损的底层看！如果能看到粗糙的砂子或石子，Material_Exposure 必须输出 Exposed_Coarse_Particles，不可遗漏！
           - 关于 Porosity：不要把起皮或剥落后露出的凹凸不平的基底误认为是孔隙！只有当看到原本平整的表面有明显的“独立小圆孔/蜂窝眼”时，才能输出孔隙率。对于大面积起皮的粗糙面，Porosity必须输出 None
           - 关于 Edge_Condition (边缘状态)：起皮的边缘通常带有断层或碎裂感，请优先考虑输出 Broken，而不是 Diffuse。
           - 附着物观测 (Surface_Attachments)：极其重要！当你观察到表面的绿色、黑褐色污渍时，请务必放大观察其纹理！如果它看起来像绒毛、青苔，必须输出 "Fuzzy_Patch"。绝对不能因为它是平面的就填 "None"！
           - 孔隙与微观凹陷观测 (Porosity & Local_Profile)： 极其重要！请务必将视线聚焦于混凝土表面的微小细节！如果你观察到表面分布着像针眼、气泡破裂留下的小圆坑，哪怕它们非常微小，也绝对不能填 "None"！ 优先在Porosity 中输出"High_Porosity"，并在 Global_Shape 中优先考虑它们是否呈现 "Circular"（圆形）
           - 局部轮廓 (Local_Profile) 微观观测指南：不要只关注宏观的大面积脱落！当你看到表面分布着由于气泡破裂留下的微小圆孔、针眼时，必须将 Local_Profile 提取为 "Small_Pits"！绝对不能因为坑洞太小而填 "None"
           - 关于材料裸露 (Material_Exposure) 的提取规范：这是一个极其关键的防混淆指令！如果表面完好，或者只有纯粹的表层小气孔（气泡），完全没有露出内部的粗糙砂子、石子或金属，你必须提取为 "No_Exposure"！绝对不允许输出 "None"！
           - 如果你观察到表面有一层非常薄、类似硬壳或粉刷层一样的物质脱落，且呈现出较硬的脆性断裂感，必须将 Local_Profile 提取为 "Brittle_Flake"（脆性薄片）！绝对不要使用 "Layered_Flake"
           - 仔细观察薄片脱落后露出的底部材质。如果底部相对平整，只有灰色的水泥基底，没有暴露出大颗粒的石头/砾石，必须将 Material_Exposure 提取为 "Exposed_Cement_Paste"（暴露水泥浆）！绝对不能因为有破损就盲目提取粗骨料（Exposed_Coarse_Particles）！
           - 立体析出物观测： 当观察到表面有向下悬垂的、立体的冰柱状或水滴凝固状物体时，Local_Profile 必须提取为 "Icicle_Shape"，且 Surface_Attachments 必须提取为 "Solidified_Drip"。极其重要：即使这些冰柱因为夹杂泥沙而呈现红褐色或黄色，也不要仅仅把它当作表面的 "Surface_Color" 污渍提取，必须优先提取其立体的物理轮廓！
        """

        try:
            resp1 = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_stage1},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=0.0
            )
            raw_stage1_output = resp1.choices[0].message.content
            baseline_extracted_attributes = extract_json_from_text(raw_stage1_output)
            # 防作弊过滤
            baseline_extracted_attributes = {k: v for k, v in baseline_extracted_attributes.items() if
                                             k in current_dict_full}
            print(f"     [全量提取结果]: 成功获取 {len(baseline_extracted_attributes)} 个特征")
        except Exception as e:
            baseline_extracted_attributes = {}
            print(f"     [警告] 阶段一特征提取发生崩溃: {e}")

        # ========================================================
        # 🌟 步骤 B：本地物理遮蔽与阶段二推理 (遍历 4 个组)
        # ========================================================
        for group_name, mask_weight in groups.items():
            print(f"  -> 执行 {group_name} ...")

            # 1. 深拷贝基准全量 JSON
            masked_attributes = copy.deepcopy(baseline_extracted_attributes)

            # 2. 本地代码实施物理遮蔽 (Ablation Logic)
            if mask_weight is not None:
                for attr in list(masked_attributes.keys()):
                    # 获取该属性在当前病害中的真实权重，默认为 0 (即 W0)
                    w = WEIGHTS.get(TARGET_CATEGORY, {}).get(attr, 0)
                    if w == mask_weight:
                        # 命中要遮蔽的权重，强制置为 "None"
                        masked_attributes[attr] = "None"

            # 3. 计算 WASC 得分 (用遮蔽后的字典去跟 GT 比对)
            was_score = calculate_wasc_with_gt(masked_attributes, actual_gt_attrs, WEIGHTS[TARGET_CATEGORY])

            # 4. 阶段二：纯文本逻辑推理
            prompt_stage2 = f"""
            你现在是一个顶级土木工程诊断专家。你无法看到现场图片。
            检验科为你送来了一份关于某个混凝土结构病害的视觉特征属性表：
            【属性报告】: {json.dumps(masked_attributes, ensure_ascii=False)}

            请严格根据上述属性报告的文本描述，运用物理逻辑推理出这是什么病害。
            候选类别包括: {ALL_CATEGORIES}

            【专家启发式诊断知识库（基于 ACI/GB 权威定义与视觉映射）】：
            请在推理时，必须严格参考以下权威物理定义与视觉属性的映射关系进行排他性推导：

            --- 1. 结构剥离与退化类 (Deterioration & Spalling) ---
            1. 【崩解 (Disintegration)】：
               - ACI CT-25 定义：Reduction into small fragments and subsequently into particles. (混凝土退化、碎裂成小碎片，并最终变成松散颗粒的现象。)
               - 视觉映射：决定性体征为松散的颗粒状附着 ("Surface_Attachments": "Granular") 伴随骨料大面积外露 ("Material_Exposure": "Exposed_Coarse_Particles")。因其属于材料缺失病害，通常伴随凹陷 ("Local_Profile": "Depression") 与破碎边缘 ("Edge_Condition": "Broken")。
            2. 【剥落 (Spalling)】：
               - ACI CT-25 定义：A fragment, usually in the shape of a flake, detached from a larger mass by a blow, the action of weather, pressure, or expansion within the larger mass(通常呈薄片状或块状的碎片，因打击、气候或压力而从较大的块体上脱落。)
               - 视觉映射：决定性核心特征是实体块的脱落，表现为明确的局部深坑或厚度差 ("Local_Profile": "Depression")。通常伴随内部骨料裸露 ("Material_Exposure": "Exposed_Coarse_Particles")。
               - 【防混淆】：若主要呈现大面积松散粉化颗粒(Granular)则优先判为崩解；若为干净的块状脱落坑则判为剥落。
            3. 【起皮 (Scaling)】：
               - ACI CT-25 定义：Local flaking or peeling away of the near-surface portion of hardened concrete. (混凝土近表面部分的局部剥落或脱皮。)
               - 视觉映射：表面呈现浅层脱落 ("Local_Profile": "Layered_Flake")，露出粗骨料颗粒（Exposed_Coarse_Particles），边缘呈现不规则的剥离状 ("Edge_Condition": "Broken")，深度与破坏力明显弱于剥落。

            --- 2. 施工与材质缺陷类 (Construction & Material Defects) ---
            4. 【冷缝 (Cold Joint)】：
               - ACI CT-25 定义：A joint or discontinuity resulting from a delay in placement of sufficient duration to preclude intermingling and bonding of the material, or where mortar or plaster rejoin or meet.(由于浇筑延迟时间过长，导致两层材料无法充分混合和结合而形成的接缝或不连续面)
               - 视觉映射：视觉形态呈线状 ("Global_Shape": "Linear")。决定性区分特征：必须位于结构接缝处 ("Relative_Position": "Joint")！若不在接缝处，不可判为冷缝。
            5. 【蜂窝 (Honeycomb)】：
               - ACI/GB 定义：Voids left in concrete between coarse aggregates due to inadequate consolidation（混凝土浇筑振捣不足，导致砂浆未能填满粗骨料之间的空隙。）
               - 视觉映射：具备显著的中高孔隙率 ("Porosity": "Moderate_Porosity" 或 "High_Porosity")，骨料外露（Exposed_Coarse_Particles），但孔洞内部无大量松散退化的碎屑。
            6. 【气孔 (Bug Holes)】：
               - ACI CT-25 定义：Small regular or irregular cavities, usually not exceeding 15 mm in diameter, resulting from entrapment of air bubbles in the surface of formed concrete (通常直径不超过15毫米的微小规则或不规则空腔，由浇筑时截留在模板表面的气泡引起)
               - 视觉映射：呈现小孔（Local_Profile：Small_Pits）的视觉特征，中高的孔隙 ("Porosity": "High_Porosity")，形态多为圆形 ("Global_Shape": "Circular")，无显著粗骨料外露（No_Exposure）。
            7. 【钢筋裸露 (Exposed Rebar)】：
               - GB 定义：混凝土保护层破坏导致内部钢筋暴露。
               - 视觉映射：决定性证据为极其明确的金属材质外露 ("Material_Exposure": "Exposed_Metal_Bar")。

            --- 3. 裂缝类 (Cracking) ---
            8. 【定向裂缝 (Directional Cracking)】：
               - ACI CT-25 定义：A complete or incomplete separation of either concrete or masonry into two or more parts produced by breaking or fracturing.（混凝土因断裂或破裂而产生的完全或不完全的分离（通常具有明确的横向、纵向或对角方向））
               - 视觉映射：呈线状 ("Global_Shape": "Linear")，无复杂的网状交织 ("Branching_Pattern": "None" )，路径多为蜿蜒（Path_Regularity：Meandering）且位置不在施工接缝处。
            9. 【网状裂缝 (Mapped Cracking)】：
               - ACI CT-25 定义：Intersecting cracks that extend below the surface of hardened concrete that vary in width from fine and barely visible to open and well-defined  (在硬化混凝土表面下延伸的相互交错的裂缝，宽度从极细微到明显张开不等。)
               - 视觉映射：决定性特征是裂缝交叉互连 ("Branching_Pattern": "Interconnected_Network"或"Simple_Branching")，外观呈网状 ("Global_Shape": "Reticulate")。

            --- 4. 渗漏与化学析出类 (Seepage & Exudation) ---
            10. 【泛碱 (Efflorescence)】：
                - ACI CT-25 定义：A generally white deposit formed when water-soluble compounds emerge in solution from concrete, masonry, or plaster substrates and precipitate by reaction such as carbonation or crystallize by evaporation (水溶性化合物以溶液形式从混凝土内部渗出，并因水分蒸发而结晶形成的通常呈白色的沉积物)
                - 视觉映射：颜色特征多为白色 ("Surface_Color": "White")，伴随结晶状附着 ("Surface_Attachments": "Crystalline")。作为表面覆盖物，通常底层轮廓边缘弥散的 ("Local_Profile": "Diffuse")。
            11. 【结壳 (Incrustation)】：
                - ACI CT-25 定义：Crusting occurs when the top surface of the concrete begins to set or dry out rapidly while the underlying concrete is still in a plastic state.(当底层混凝土仍处于塑性状态时，顶层表面因快速变干或凝结而形成的脆弱硬壳)
                - 视觉映射：表面有明显的“脆性硬壳” ("Local_Profile": "Brittle_Flake")，且露出平滑的水泥浆体（Material_Exposure：Exposed_Cement_Paste）。
            12. 【钟乳石状析出 (Stalactite)】：
                - ACI CT-25 定义：A downward-pointing deposit formed as an accretion of mineral matter produced by evaporation of dripping water from the surface of concrete, commonly shaped like an icicle (由混凝土表面滴水的蒸发所产生的矿物质附着而形成的向下悬垂类似冰柱状的沉积物。)
                - 视觉映射：决定性特征固体滴落状（"Surface_Attachments"："Solidified_Drip"），且多出现在天花板处呈现冰柱状("Surface_Orientation"："Ceiling_Surface", "Local_Profile": "Icicle_Shape")，多为白色（"Surface_Color": "White"）。

            --- 5. 表面污渍类 (Stains & Discoloration) ---
            13. 【水渍 (Water Stain)】：
                - GB 50108-2008 定义：混凝土表面无明显水珠滴落，但因水分蒸发后留下的色泽改变和明显的边界轮廓。
                - 视觉映射：核心特征为大面积或不规则的色泽改变,多数会出现“咖啡环效应” ,且有明显轮廓("Stain_Pattern": "Coffee_Ring"，"Surface_Color"："Brown"或"Dark"，"Edge_Condition"："Sharp")。
                - 【防混淆警告】：在真实工程中，水渍常因夹杂泥沙而呈现褐色。如果 "Surface_Color" 为 "Brown" 或 "Reddish_Brown"，但形态呈现中间浅，四周深的“咖啡环效应”而非线状流挂，绝对优先判定为【水渍】！
            14. 【铁锈 (Rust Stain)】：
                - ACI/GB 定义：由于内部金属/钢筋锈蚀，铁氧化物随水分渗出，在混凝土表面沉积形成的红褐色或黄褐色污渍
                - 视觉映射：颜色必须为 ("Surface_Color": "Reddish_Brown")。
            15. 【生物生长污渍 (Biological Stain)】：
                - ACI 定义：Colonization of damp concrete surfaces by living organisms such as algae, fungi, moss, or lichen, typically presenting as green, black, or brown pigmented patches or fuzzy deposits.
                - 视觉映射：颜色通常为 ("Surface_Color": "Green")，表面附着物多呈现为绒毛状（"Surface_Attachments"："Fuzzy_Patch"），分布多呈斑块状分布 ("Stain_Pattern": "Patchy")。
            【专家思维链降噪规则】：
            在匹配上述视觉映射时，请注意土木病害的特征权重是不平等的。当属性之间发生指向性冲突时（例如形状特征指向 A 病害，颜色/附着物特征指向 B 病害），请遵循以下专家权重逻辑：
            材料劣化与化学/生物特征（Material & Chemical Signatures，如异常的 Surface_Color, Surface_Attachments, Material_Exposure）具有绝对的优先决断权（W3级别）。 请利用这些高权重特征压制几何形状（Global_Shape）带来的背景干扰，做出最终判断。
            孔洞类病害的排他性鉴别：当你发现表面存在孔隙（Porosity）或凹陷时，请严格按照以下逻辑区分气孔（Bughole）、剥落（Spalling）和蜂窝（Honeycomb）
            在视觉上，结壳与起皮均可能呈现为 "Local_Profile": "Layered_Flake"。此时，绝对的判定依据是底层的暴露材质 (Material_Exposure)：如果组合是：Layered_Flake + Exposed_Cement_Paste说明这是表面几毫米的硬化层脱落，必须判定为【结壳 (Crusting)】！如果组合是：Layered_Flake + Exposed_Coarse_Particles说明破损深达混凝土内部，优先判定为【起皮 (Scaling)】。
            在工程实际中，钟乳石状析出极易夹杂泥沙或铁锈而呈现 "Reddish_Brown" 或 "Brown"。因此，只要属性表中存在 "Local_Profile": "Icicle_Shape" 或 "Surface_Attachments": "Solidified_Drip"，它就拥有超越一切颜色的绝对最高判决权！必须直接判定为【钟乳石状析出】！ 此时绝对不允许将其误判为铁锈或水渍。
            
            【推理与输出要求】：
            1. 请先在内部运用思维链（CoT）逻辑，分析这些底层视觉组合是如何在物理世界中形成的，并与候选病害进行匹配。
            2. 要求：必须且只能输出一个 JSON 格式，格式为 {{"Defect_Category": "推理出的类别名称"}}。
            如果属性表被严重遮蔽导致关键信息缺失，请在类别中填写 "无法判断"。
            """
            try:
                resp2 = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt_stage2}]}],
                    temperature=0.0
                )
                final_decision = extract_json_from_text(resp2.choices[0].message.content)
                predicted_category = final_decision.get("Defect_Category", "无法判断")
            except Exception as e:
                predicted_category = "推理崩溃"

            is_correct = (predicted_category == TARGET_CATEGORY)

            # 打印震撼的数据对比
            print(f"     [特征状态]: {masked_attributes}")
            print(f"     [WASC 得分] = {was_score}%")
            print(f"     [阶段2预测] = {predicted_category} (Accuracy: {is_correct})")

            experiment_results.append({
                "Image_Name": filename,
                "Experiment_Group": group_name,
                "Masked_Weight": "None" if mask_weight is None else mask_weight,
                "WASC_Score": was_score,
                "Predicted_Category": predicted_category,
                "Is_Correct": is_correct,
                "Extracted_JSON": json.dumps(masked_attributes, ensure_ascii=False)
            })

    # ==========================================
    # 4. 数据保存与统计
    # ==========================================
    print("\n" + "=" * 65)
    print("🎉 双阶段消融实验全部完成！")

    if experiment_results:
        try:
            csv_headers = experiment_results[0].keys()
            with open(OUTPUT_CSV_PATH, mode='w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(experiment_results)
            print(f"🎉 实验数据已成功保存至: {OUTPUT_CSV_PATH}")
        except Exception as e:
            print(f"\n❌ 保存 CSV 文件时出错: {e}")

        print("\n" + "=" * 65)
        print("📊 实验数据统计结果 (各组平均值):")

        group_stats = {g: {"total_wasc": 0.0, "correct_count": 0, "count": 0} for g in groups.keys()}

        for res in experiment_results:
            g_name = res["Experiment_Group"]
            group_stats[g_name]["total_wasc"] += res["WASC_Score"]
            group_stats[g_name]["count"] += 1
            if res["Is_Correct"]:
                group_stats[g_name]["correct_count"] += 1

        for g_name, stats in group_stats.items():
            if stats["count"] > 0:
                avg_wasc = stats["total_wasc"] / stats["count"]
                avg_acc = (stats["correct_count"] / stats["count"]) * 100
                print(f"  👉 [{g_name}] : 平均 WASC 得分 = {avg_wasc:.2f}%  |  平均分类准确率 = {avg_acc:.2f}%")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    run_two_stage_experiment()