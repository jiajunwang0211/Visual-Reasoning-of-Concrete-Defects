# -*- coding: utf-8 -*-
import os
import json
import base64
import re
from openai import OpenAI
import csv

# ==========================================
# 1. 配置区域 (请填入你的真实信息)
# ==========================================
API_KEY = ""
BASE_URL = ""
MODEL_NAME = "gpt-4o"

# 🌟 核心修改：你只需要指定这一个总目录即可！
VAL_ROOT_DIR = r"C:\Users\admin\Desktop\15\data_final_noisy"
OUTPUT_CSV_PATH = r"C:\Users\admin\Desktop\小论文文献\实验\实验2_val_noisy.csv"
ONTOLOGY_PATH = r"C:\Users\admin\Desktop\15\第三轮\step3_defect_refined_output\step3_defect_refined_ontology（修改）.json"

ALL_CATEGORIES = ["定向裂缝", "网状裂缝", "钢筋裸露", "剥落", "起皮", "崩解", "蜂窝", "水渍", "铁锈", "生物生长污渍","泛碱", "结壳", "钟乳石状析出", "气孔", "冷缝"]
TEST_FOLDERS = ALL_CATEGORIES

# 你的真实权重字典 (WASC 算分引擎)
WEIGHTS = {
    "定向裂缝": {"Global_Shape": 3, "Feature_Orientation": 1, "Branching_Pattern": 3, "Path_Regularity": 1},
    "钟乳石状析出": {"Surface_Attachments": 3, "Surface_Orientation": 3, "Surface_Color": 1, "Local_Profile": 3},
    "网状裂缝": {"Global_Shape": 3, "Branching_Pattern": 3, "Path_Regularity": 1},
    "蜂窝": {"Material_Exposure": 3, "Surface_Texture": 3, "Porosity": 3, "Local_Profile": 1},
    "钢筋裸露": {"Material_Exposure": 3, "Local_Profile": 1},
    "冷缝": {"Path_Regularity": 1, "Relative_Position": 3, "Global_Shape": 3, "Local_Profile": 1},
    "崩解": {"Surface_Attachments": 3, "Material_Exposure": 3, "Edge_Condition": 3, "Local_Profile": 1},
    "剥落": {"Local_Profile": 3, "Material_Exposure": 3, "Edge_Condition": 1, "Surface_Texture": 1},
    "泛碱": {"Surface_Color": 3, "Surface_Attachments": 3, "Edge_Condition": 1},
    "水渍": {"Surface_Color": 3, "Edge_Condition": 3, "Stain_Pattern": 3, "Global_Shape": 1},
    "起皮": {"Local_Profile": 3, "Material_Exposure": 3, "Edge_Condition": 1, "Surface_Texture": 1},
    "生物生长污渍": {"Surface_Color": 3, "Surface_Attachments": 3, "Stain_Pattern": 1},
    "气孔": {"Local_Profile": 3, "Material_Exposure": 3, "Porosity": 1, "Edge_Condition": 1},
    "结壳": {"Local_Profile": 3, "Material_Exposure": 3, "Edge_Condition": 1},
    "铁锈": {"Surface_Color": 3, "Stain_Pattern": 1}
}

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ------------------------------------------
# 加载专家本体字典
# ------------------------------------------
try:
    with open(ONTOLOGY_PATH, 'r', encoding='utf-8') as f:
        full_ontology = json.load(f)
    ONTOLOGY_COMBINED = {}
    ONTOLOGY_COMBINED.update(full_ontology.get("S_int", {}))
    ONTOLOGY_COMBINED.update(full_ontology.get("S_ctx", {}))
except Exception as e:
    print(f"❌ 读取本地本体字典失败: {e}")
    ONTOLOGY_COMBINED = {}

current_dict_full = {}
for attr, details in ONTOLOGY_COMBINED.items():
    current_dict_full[attr] = {"定义": details["definition"], "可选值": details["visual_anchors"]}


# ==========================================
# 2. 辅助函数
# ==========================================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_json_from_text(text):
    try:
        cleaned_text = text.replace('```json', '').replace('```', '').strip()
        match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}
    except:
        return {}


def calculate_wasc_with_gt(predicted_attrs, ground_truth, weights):
    """基于权重的算分器 (充当置信度引擎)"""
    total_score = 0.0
    max_possible_score = sum([weights.get(k, 0) for k in ground_truth.keys()])
    if max_possible_score == 0:
        return 0.0

    for attr, true_val in ground_truth.items():
        pred_val = predicted_attrs.get(attr)

        # 🌟 修复：只拦截真正的空对象或空字符串，把字符串 "None" 从拦截黑名单中踢出去！
        if pred_val is None or pred_val == "" or pred_val == "Unknown":
            continue

        is_correct = False
        if isinstance(true_val, list):
            if pred_val in true_val: is_correct = True
        else:
            if pred_val == true_val: is_correct = True

        # 如果匹配成功（包括 pred_val 和 true_val 都是 "None" 的情况），予以加分！
        if is_correct:
            total_score += weights.get(attr, 0)

    return round((total_score / max_possible_score) * 100, 2)


# ==========================================
# 3. 实验二主程序
# ==========================================
def run_full_experiment_2_with_wasc():
    print(f"\n初始化【实验二：大模型+专家字典测试 (带WASC置信度)】...")
    print("=" * 65)

    if not os.path.exists(VAL_ROOT_DIR):
        print(f"❌ 找不到验证集目录: {VAL_ROOT_DIR}")
        return

    experiment_results = []
    total_correct = 0
    total_images = 0

    # 外层循环：遍历 15 个病害类别文件夹
    for category_folder in os.listdir(VAL_ROOT_DIR):
        category_path = os.path.join(VAL_ROOT_DIR, category_folder)
        if not os.path.isdir(category_path) or category_folder not in TEST_FOLDERS:
            continue

        true_category = category_folder

        # ----------------------------------------------------
        # 🌟 核心修改：进入文件夹后，先找到并读取唯一的 GT JSON 文件
        # ----------------------------------------------------
        category_gt = {}
        # 找出当前文件夹下所有的 .json 文件
        json_files = [f for f in os.listdir(category_path) if f.lower().endswith('.json')]

        if json_files:
            # 假设每个类别文件夹里只有 1 个汇总的 json，直接读第一个
            json_path = os.path.join(category_path, json_files[0])
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    category_gt = json.load(jf)
                print(f"\n✅ 成功加载【{true_category}】的基准文件: {json_files[0]}")
            except Exception as e:
                print(f"\n⚠️ 读取 {json_files[0]} 失败: {e}")
        else:
            print(f"\n⚠️ 【{true_category}】文件夹下找不到 .json 文件，图片置信度将为 N/A")

        # 过滤出所有图片文件
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # image_files = image_files[:5]

        print(f"📁 开始测试类别: 【{true_category}】 (共 {len(image_files)} 张图片)")

        # 内层循环：测试每张图片
        for filename in image_files:
            total_images += 1
            image_path = os.path.join(category_path, filename)
            base64_image = encode_image(image_path)

            # 🌟 核心修改：直接用图片全名（如 "Cold Joint_1.jpg"）作为 Key 去刚才读的字典里查！
            raw_gt = category_gt.get(filename, {})

            # 解析出真正的 GT 属性字典 (合并 S_int 和 S_ctx)
            actual_gt_attrs = {}
            if raw_gt:
                if 'S_int' in raw_gt: actual_gt_attrs.update(raw_gt['S_int'])
                if 'S_ctx' in raw_gt: actual_gt_attrs.update(raw_gt['S_ctx'])
                if not actual_gt_attrs: actual_gt_attrs = raw_gt  # 兼容没有嵌套的情况

            # --- 🌟 阶段一：特征提取 ---
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
            - 【极其重要：自然裂缝 vs 施工接缝(冷缝)的排他性区分】：
              1. 走势判断：如果缝隙路径是像树枝、闪电一样曲折蔓延的自然开裂，请将 `Path_Regularity` 提取为 "Meandering"，此时 `Relative_Position` 必须保持 "None"！
              2. 提取 Relative_Position 时：只有当你清晰看到线状痕迹两侧存在材质、颜色或浇筑批次的明显差异，或具备平直切割特征时（哪怕这条缝是否开裂、是否漏水），必须提取 "Joint"。对于普通的不规则开裂，必须输出 "None"。
              3. 形状的诚实提取 (Global_Shape)：对于接缝处，请如实提取其形状！如果接缝边缘平整，提取为 "Linear"；如果接缝边缘因为掉渣、起皮变得坑坑洼洼，请如实提取为 "Irregular"！不要因为是接缝就强行提取线状。
            - 【微小孔洞的线性排异】：如果画面中出现了一排直线排列的特征，请放大看！如果是沿着模板边缘排列的独立、微小的气泡坑洞群，必须提取 "Local_Profile": "Small_Pits"，且 "Relative_Position" 必须保持 "None"！绝对不允许因为它们排列成直线就将其误判为 "Joint"（接缝）！
            - 关于 "Surface_Attachments" (表面附着物) 与 "Local_Profile" (局部轮廓)：
              1. 请极其严格地区分“二维的颜色印记(2D)”和“三维的物理凸起/附着物(3D)”。
              2. 如果只是混凝土表面颜色的改变（如变暗、变黄、变白），用手摸是平的，没有任何凸出于表面的粉末、结晶或石块，那么 "Local_Profile" 必须是 "None"，"Surface_Attachments" 必须是 "None"。
              3. 绝对不要仅仅因为颜色改变或者位置在天花板上，就凭空想象出附着物或凸起。
              4. 请极其注意表面的“白色物质”。只要你在图像中看到混凝土表面有白色的粉末状、霜状、盐霜、或是硬化的白色结晶斑块，即使它们非常薄或者紧贴表面，也**必须强制提取为 "Crystalline"（结晶附着物）**。绝对不能输出 "None"。
              5. 当你在天花板（Ceiling）处看到白色的线状或斑块状物质时，请仔细观察其是否有向外凸起或向下悬垂的体积感。即使不明显，只要感觉它不是完全平贴在墙面上，就请优先提取 "Surface_Attachments": "Solidified_Drip" 或 "Local_Profile": "Icicle_Shape"
            - 颜色观测：如果表面有明显的白色覆盖物或析出物，Surface_Color 必须如实提取为 "White"。当你提取 Surface_Color 时，绝对不允许输出周围正常混凝土/白墙的背景色（如 White, Light Gray）。你必须更加关注发生变色、水晕、沉积的那条边缘或斑块本身的颜色，优先提取。
            - 在判断暴露物以及孔隙时，请务必谨慎：
                只有当你在脱落区域清晰地看到完整的、大颗粒的碎石/卵石轮廓时，才能提取 Exposed_Coarse_Particles。
                如果脱落区域只是颜色较深、表面粗糙、或者带有沙粒质感，但没有明显的大块石头裸露，请必须提取为 Exposed_Cement_Paste（水泥浆）。千万不要把沾有泥沙的粗糙水泥面误认为粗骨料！
                如果你看到的是密集、相对平滑的针尖状或米粒大小的小坑洞，请将其 Local_Profile 提取为 Small_Pits。对于此类表面病害，其内部通常是灰色的水泥浆体，千万不要将孔洞本身的阴影或粗糙感误判为粗骨料，Material_Exposure 应优先提取为No_Exposure
            - 附着物观测：如果是白色的粉末或结晶，提取为 "Crystalline" 
            - 【崩解特征的极其强制捕获 (Surface_Attachments)】：这是区分剥落和崩解的唯一生命线！绝大多数多模态模型会漏掉这一点！
              1. 不要死板地只去寻找“飘落的粉末”。请极其仔细地观察破损区域的【砂浆基体】！
              2. 只要基体看起来像“受潮的饼干”一样呈现酥脆、烂糟糟、风化、或是骨料周围的砂浆严重流失导致骨料摇摇欲坠的【材质退化感】，哪怕你只看到了粗骨料外露而没有看到粉末，也**必须强制提取 "Surface_Attachments": "Granular"**！
              3. 只有当坑洞像是被“外力硬生生砸出来”的，内部相对干净、坚硬，没有任何酥松风化感时，Surface_Attachments 才能填 "None"。
            - 【深坑(Depression) vs 浅层起皮(Layered_Flake)】：
              1. 如果破损面积较大，但深度非常均匀且极浅（只有表皮砂浆剥落，露出下面的石子但石子没有掉），必须输出 "Layered_Flake"！
              2. 只有当破损具有明显的“深度落差”，像被硬生生砸掉或抠掉了一个带有厚度的“实体块”，才能输出 "Depression"！
              3. 请极其注意：只要破损区域的边缘存在明显的厚度断层、阴影，或者暴露出大量的粗骨料（石头），请绝对禁止使用 Layered_Flake，必须强制提取为 Depression
            - 轮廓观测：只有当表面存在边界清晰的实体凹陷、厚度差或坑洞时，才可选 "Depression"。对于纯粹的表面颜色变化或污渍，Local_Profile 必须保持 "None"。
            
            - 【极其重要：水渍的“咖啡环效应 (Coffee Ring)”强制捕获】：
              当观察到混凝土表面有深色、褐色或黄色污渍时，请不要急着填 "Blotchy" (斑驳)！你必须极其仔细地观察污渍的【边缘颜色梯度】！
              如果污渍呈现出“四周边缘有一圈颜色更深、更明显的轮廓线，而污渍内部中心区域颜色相对较浅”的现象（即水分蒸发留下的典型痕迹），你**必须强制提取 `Stain_Pattern`: "Coffee_Ring"**！
              绝对禁止用 "Blotchy" 来敷衍概括这种具有明确边缘梯度的蒸发痕迹！只有当污渍真的是毫无规律、颜色深浅完全随机分布的脏污时，才允许使用 "Blotchy"。
            
            - 【深坑(Depression) vs 浅层起皮(Layered_Flake) 的排他性提取】：
              请极其仔细地评估破损区域的“深度与体积”！
              1. 只有当破损仅仅是表面几毫米的砂浆像墙皮一样极薄地剥离脱落（厚度如纸或指甲盖），才能输出 "Layered_Flake"！
              2. 只要连带内部结构掉落，形成了一个能明显看出“缺失了一个实体块、有明显阴影或深度差”的坑洞或深槽，即使面积不大，也绝对不能用 Layered_Flake，必须果断提取为 "Depression"！
              3. 注意！只要有坑洞脱落，请务必仔细看底部！通常都会暴露出粗糙的石子或粗骨料，此时必须提取为 "Exposed_Coarse_Particles"，不可遗漏！

            - 边缘状态 (Edge_Condition)：起皮的边缘通常带有断层或碎裂感，请优先考虑输出 Broken，而不是 Diffuse。
            - 孔隙与破损的区别 (Porosity)：不要把起皮或剥落后露出的凹凸不平的基底误认为是孔隙！只有当看到原本平整的表面有明显的“独立小圆孔/蜂窝眼”时，才能输出孔隙率。对于大面积起皮的粗糙面，Porosity必须输出 None。在评估 Porosity 时，请特别注意骨料之间是否存在连续的、成簇的微小空洞或凹陷。如果发现表面像马蜂窝一样存在密集分布的小孔洞，必须标记为 High_Porosity。
            - 生物附着物 (Surface_Attachments)：极其重要！当你观察到表面的绿色、黑褐色污渍时，请务必放大观察其纹理！如果它看起来像绒毛、青苔，必须输出 "Fuzzy_Patch"。绝对不能因为它是平面的就填 "None"！
            - 微观气孔观测 (Porosity & Local_Profile)： 极其重要！请务必将视线聚焦于混凝土表面的微小细节！如果你观察到表面分布着像针眼、气泡破裂留下的小圆坑，哪怕它们非常微小，也绝对不能填 "None"！必须将 Local_Profile 提取为 "Small_Pits"，且在 Porosity 中优先输出 "High_Porosity"。并在 Global_Shape 中优先考虑它们是否呈现 "Circular"（圆形）。
            - 材料裸露防混淆 (Material_Exposure)：如果表面完好，或者只有纯粹的表层小气孔（气泡），完全没有露出内部的粗糙砂子、石子或金属，你必须提取为 "No_Exposure"！绝对不允许输出 "None"！
            - 立体析出物观测： 当观察到表面有向下悬垂的、立体的冰柱状或水滴凝固状物体时，Local_Profile 必须提取为 "Icicle_Shape"，且 Surface_Attachments 必须提取为 "Solidified_Drip"。极其重要：即使这些冰柱因为夹杂泥沙而呈现红褐色或黄色，也不要仅仅把它当作表面的 "Surface_Color" 污渍提取，必须优先提取其立体的物理轮廓！
            - 【极其重要：Honeycombed (蜂窝纹理) 的严格前置条件】：请极其谨慎地提取 "Honeycombed"！绝对不能仅仅因为表面有密集的坑洼或气泡眼就选择它。
              1. 触发条件：提取 "Honeycombed" 必须严格满足一个物理前提——你必须清晰地看到【粗骨料（石头）大量裸露】，并且石头之间缺乏砂浆包裹形成了深层空洞。
              2. 气孔分流：如果表面像马蜂窝一样布满密集的小圆坑（气泡孔），但底层依然是灰色的水泥浆体，**没有露出内部粗糙的石头**，那么 `Surface_Texture` **绝对不允许**选择 "Honeycombed"（请降级为 "Rough" 等其他材质），这类表层孔洞特征必须且只能交由 `Local_Profile` 的 "Small_Pits" 和 `Porosity` 的中高孔隙率去表达！
            """
            try:
                resp1 = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt_stage1},
                                                           {"type": "image_url", "image_url": {
                                                               "url": f"data:image/jpeg;base64,{base64_image}"}}]}],
                    temperature=0.0
                )
                stage1_attrs = extract_json_from_text(resp1.choices[0].message.content)
                stage1_attrs = {k: v for k, v in stage1_attrs.items() if k in current_dict_full}
            except:
                stage1_attrs = {}

            # 🚀 算分环节：计算 WASC Confidence Score
            if actual_gt_attrs and true_category in WEIGHTS:
                wasc_score = calculate_wasc_with_gt(stage1_attrs, actual_gt_attrs, WEIGHTS[true_category])
            else:
                wasc_score = "N/A"

            # --- 🌟 阶段二：推理 ---
            prompt_stage2 = f"""
            你现在是一个顶级土木工程诊断专家。你无法看到现场图片。
            检验科为你送来了一份关于某个混凝土结构病害的视觉特征属性表：
            【属性报告】: {json.dumps(stage1_attrs, ensure_ascii=False)}

            请严格根据上述属性报告的文本描述，运用物理逻辑推理出这是什么病害。
            候选类别包括: {ALL_CATEGORIES}

            【专家启发式诊断知识库（基于 ACI/GB 权威定义与视觉映射）】：
            请在推理时，必须严格参考以下权威物理定义与视觉属性的映射关系进行排他性推导：

            --- 1. 结构剥离与退化类 (Deterioration & Spalling) ---
            1. 【崩解 (Disintegration)】：
               - ACI CT-25 定义：Reduction into small fragments and subsequently into particles. (混凝土退化、碎裂成小碎片，并最终变成松散颗粒的现象。)
               - 视觉映射：决定性体征为松散的颗粒状附着 ("Surface_Attachments": "Granular") 伴随骨料大面积外露 ("Material_Exposure": "Exposed_Coarse_Particles")。因其属于材料缺失病害，通常伴随凹陷 ("Local_Profile": "Depression") 与破碎边缘 ("Edge_Condition": "Broken")。
               - 核心锚点：存在明显的松散颗粒或粉化退化现象 ("Surface_Attachments": "Granular")，则判定为【崩解】
            2. 【剥落 (Spalling)】：
               - ACI CT-25 定义：A fragment, usually in the shape of a flake, detached from a larger mass by a blow, the action of weather, pressure, or expansion within the larger mass(通常呈薄片状或块状的碎片，因打击、气候或压力而从较大的块体上脱落。)
               - 视觉映射：决定性核心特征是实体块的脱落，表现为明确的局部深坑或厚度差 ("Local_Profile": "Depression")。通常伴随内部骨料裸露 ("Material_Exposure": "Exposed_Coarse_Particles")。
               - 绝对判定底线：只要属性表中明确提取到了 "Depression"（具有体积感和深度的深坑），并且 "Material_Exposure" 包含 "Exposed_Coarse_Particles" (粗骨料外露)！此时，无论边缘状态如何，【起皮 (Scaling)】必须立刻出局！如果伴随 "Granular" 则判为【崩解】；如果没有 "Granular"，则绝对且只能判定为【剥落】！
            3. 【起皮 (Scaling)】：
               - ACI CT-25 定义：Local flaking or peeling away of the near-surface portion of hardened concrete. (混凝土近表面部分的局部剥落或脱皮。)
               - 视觉映射：表面呈现浅层脱落 ("Local_Profile": "Layered_Flake")，露出粗骨料颗粒（Exposed_Coarse_Particles），边缘呈现不规则的剥离状 ("Edge_Condition": "Broken")，深度与破坏力明显弱于剥落。
               - 起皮与剥落的唯一界限在于深度！只要 Local_Profile 是 "Layered_Flake"，无论露没露骨料，都必须优先判为【起皮】；一旦提取为 "Depression" (实体深坑)，起皮立刻出局，必须在剥落或崩解中选择！
               - 专家备忘：起皮不仅会露出粗骨料，也可能仅发生于表层砂浆导致只露出水泥浆 ("Exposed_Cement_Paste")。与结壳不同，当起皮仅露水泥浆时，其底面通常带有后期风化的粗糙感 ("Surface_Texture": "Rough")。

            --- 2. 施工与材质缺陷类 (Construction & Material Defects) ---
            4. 【冷缝 (Cold Joint)】：
               - ACI CT-25 定义：A joint or discontinuity resulting from a delay in placement of sufficient duration to preclude intermingling and bonding of the material, or where mortar or plaster rejoin or meet.(由于浇筑延迟时间过长，导致两层材料无法充分混合和结合而形成的接缝或不连续面)
               - 视觉映射：视觉形态呈线状 ("Global_Shape": "Linear")。决定性区分特征：必须位于结构接缝处 ("Relative_Position": "Joint")！若不在接缝处，不可判为冷缝。
            5. 【蜂窝 (Honeycomb)】：
               - ACI/GB 定义：Voids left in concrete between coarse aggregates due to inadequate consolidation（混凝土浇筑振捣不足，导致砂浆未能填满粗骨料之间的空隙。）
               - 视觉映射：具备显著的中高孔隙率 ("Porosity": "Moderate_Porosity" 或 "High_Porosity")，骨料外露（Exposed_Coarse_Particles），且呈现蜂窝状（Honeycomb）但孔洞内部无大量松散退化的碎屑。
            6. 【气孔 (Bug Holes)】：
               - ACI CT-25 定义：Small regular or irregular cavities, usually not exceeding 15 mm in diameter, resulting from entrapment of air bubbles in the surface of formed concrete (通常直径不超过15毫米的微小规则或不规则空腔，由浇筑时截留在模板表面的气泡引起)
               - 视觉映射：呈现小孔（Local_Profile：Small_Pits）的视觉特征，中高的孔隙 ("Porosity": "High_Porosity")，无显著粗骨料外露（No_Exposure）。
               - 专家备忘：即使此时 Surface_Texture 显示为 "Honeycombed" 或具有高孔隙率，它也仅仅代表“密集的表层气孔簇”，只要底层没露出粗糙石子，依然是气孔！若气孔沿施工缝排列 (Relative_Position: Joint)，这也是气孔，而非冷缝。
            7. 【钢筋裸露 (Exposed Rebar)】：
               - GB 定义：混凝土保护层破坏导致内部钢筋暴露。
               - 视觉映射：决定性证据为极其明确的金属材质外露 ("Material_Exposure": "Exposed_Metal_Bar")。

            --- 3. 裂缝类 (Cracking) ---
            8. 【定向裂缝 (Directional Cracking)】：
               - ACI CT-25 定义：A complete or incomplete separation of either concrete or masonry into two or more parts produced by breaking or fracturing.（混凝土因断裂或破裂而产生的完全或不完全的分离（通常具有明确的横向、纵向或对角方向））
               - 视觉映射：呈线状 ("Global_Shape": "Linear")，无复杂的网状交织 ("Branching_Pattern": "None" )，路径多为蜿蜒（Path_Regularity：Meandering）且位置不在施工接缝处。
            9. 【网状裂缝 (Mapped Cracking)】：
               - ACI CT-25 定义：Intersecting cracks that extend below the surface of hardened concrete that vary in width from fine and barely visible to open and well-defined  (在硬化混凝土表面下延伸的相互交错的裂缝，宽度从极细微到明显张开不等。)
               - 视觉映射：决定性特征是裂缝交叉互连 ("Branching_Pattern": "Interconnected_Network"或"Simple_Branching")，外观呈网状或者不规则 ("Global_Shape": "Reticulate"或"Irregular")，路径多为蜿蜒（Path_Regularity：Meandering）。

            --- 4. 渗漏与化学析出类 (Seepage & Exudation) ---
            10. 【泛碱 (Efflorescence)】：
                - ACI CT-25 定义：A generally white deposit formed when water-soluble compounds emerge in solution from concrete, masonry, or plaster substrates and precipitate by reaction such as carbonation or crystallize by evaporation (水溶性化合物以溶液形式从混凝土内部渗出，并因水分蒸发而结晶形成的通常呈白色的沉积物)
                - 视觉映射：颜色特征多为白色 ("Surface_Color": "White")，伴随结晶状附着 ("Surface_Attachments": "Crystalline")。作为表面覆盖物，通常底层轮廓边缘弥散的 ("Edge_Condition": "Diffuse")。
            11. 【结壳 (Incrustation)】：
                - ACI CT-25 定义：Crusting occurs when the top surface of the concrete begins to set or dry out rapidly while the underlying concrete is still in a plastic state.(当底层混凝土仍处于塑性状态时，顶层表面因快速变干或凝结而形成的脆弱硬壳)
                - 视觉映射：同样表现为浅层脱落 ("Local_Profile": "Layered_Flake")，但决定性区分特征是：脱落后底层只露出平滑的水泥浆体（"Material_Exposure": "Exposed_Cement_Paste"），绝不露出粗糙骨料。
                - 专家备忘：结壳是早期塑性阶段的病害，底面未完全水化，具有相对平滑的质感 ("Surface_Texture": "Smooth")，绝不会露出粗骨料。
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

            【绝对拦截与优先级截流规则（严格按顺序执行）】：
            当属性特征发生冲突时，必须按照以下 1->2->3->4 的最高优先级进行强制截流：
                   
            优先级 1: 【接缝绝对霸权 (Joint)】
            只要 `Relative_Position` 提取为 "Joint" 且 `Global_Shape` 包含 "Linear"：
            -> [强制截流]：这代表缺陷发生在两次浇筑的施工缝处。此时，无论它伴随了多么严重的起皮 ("Layered_Flake")、或者是接缝处漏浆导致的蜂窝 ("High_Porosity", "Honeycombed")，都必须将其视为“接缝处伴生缺陷”。必须优先且唯一判定为【冷缝 (Cold Joint)】！绝对禁止在此类情况下判定为蜂窝或起皮！
            
            优先级 2: 【气孔的微观防线 (Small Pits)】
            IF `Local_Profile` 包含 "Small_Pits"，且 `Material_Exposure` 为 "No_Exposure" (未露出深层粗骨料)：
            -> [强制截流]：这是表面气泡未排出的典型特征。即使它呈现为线性 ("Linear")，或者纹理被误识别为 "Honeycombed"，也必须且只能判定为【气孔 (Bug Holes)】！绝对禁止误判为冷缝或蜂窝！
            
            优先级 3: 【铁锈与水渍的防伪验证 (Reddish_Brown)】
            当 `Surface_Color` 提取为 "Reddish_Brown" 时：
            1. IF 伴随 `Stain_Pattern` 为 "Coffee_Ring"（咖啡环/水晕边缘）：说明这是夹杂泥沙的脏水蒸发，判定为【水渍 (Water Stain)】。
            2. IF `Stain_Pattern` 为 "Dripping"（垂直流挂）或”Blotchy“ 或 `Global_Shape` 为 "Linear"（伴随锈胀裂缝）：即使 `Material_Exposure` 为 "No_Exposure"（未露出钢筋），也必须绝对判定为【铁锈 (Rust Stain)】！因为锈水会从内部渗出。
            3. IF `Material_Exposure` 明确为 "Exposed_Metal_Bar"（露出金属）：无条件强制判定为【铁锈 (Rust Stain)】。
            
            优先级 4: 【蜂窝与崩解的深层区分】
            IF `Material_Exposure` 明确为 "Exposed_Coarse_Particles" (深层粗骨料外露)：
            - 若伴随 "Surface_Attachments": "Granular" -> 强制判定【崩解 (Disintegration)】。
            - 若无 "Granular"，且伴随 "High_Porosity" 或 "Honeycombed" -> 强制判定【蜂窝 (Honeycomb)】。禁止判定为剥落。

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
                pred_json = extract_json_from_text(resp2.choices[0].message.content)
                predicted_category = pred_json.get("Defect_Category", "无法判断")
            except:
                predicted_category = "推理崩溃"

            # 记录并打印结果
            is_correct = (predicted_category == true_category)
            if is_correct: total_correct += 1

            mark = "✅" if is_correct else "❌"
            print(f"  {mark} {filename} | 预测: {predicted_category} | 置信度(WASC): {wasc_score}")
            if not is_correct:
                print(f"      🔍 [错判特征剖析]: {stage1_attrs}")

            experiment_results.append({
                "Image_Name": filename,
                "True_Category": true_category,
                "Predicted_Category": predicted_category,
                "Is_Correct": is_correct,
                "WASC_Confidence_Score": wasc_score,
                "Stage1_Extracted_Attributes": json.dumps(stage1_attrs, ensure_ascii=False)
            })

    # --- 结果保存 ---
    if experiment_results:
        with open(OUTPUT_CSV_PATH, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=experiment_results[0].keys())
            writer.writeheader()
            writer.writerows(experiment_results)
        print(f"\n带有置信度的数据已保存至: {OUTPUT_CSV_PATH}")
        print(f"实验总体准确率: {(total_correct / total_images) * 100:.2f}%")


if __name__ == "__main__":
    run_full_experiment_2_with_wasc()