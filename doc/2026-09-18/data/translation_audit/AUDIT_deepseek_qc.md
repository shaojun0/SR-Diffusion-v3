# 全量译文质检报告（DeepSeek deepseek-flash）

评委：`deepseek-flash`（`reasoning_effort=none`），temperature=0，逐条判定
校准表现（22 条人工标注集）：抓错率 0.93、误报率 0.00（21/22）

## 1. 总览

| 项 | 值 |
|---|---|
| 复核单元总数 | **21329** |
| 有效判决 | 21329 |
| 解析/接口失败 | 15 |
| ok | 16830（78.91%） |
| minor | 0（0.00%） |
| **major（硬错误）** | **4499（21.09%）** |

> 评委在 22 条校准集上抓错率 0.93 ⇒ 真实硬错误率约为实测值的 1/0.93 ≈ 1.08 倍。

## 2. 错误类型分布（可多标签）

| 类型 | 次数 |
|---|---:|
| term | 3426 |
| garble | 914 |
| neg | 811 |
| omit | 736 |
| extra | 352 |
| num | 240 |

## 3. 按轮次 / 语料形态

| 维度 | 单元数 | major | minor | major 占比 |
|---|---:|---:|---:|---:|
| kind:enum | 164 | 3 | 0 | 1.83% |
| kind:free | 21165 | 4496 | 0 | 21.24% |
| round:round1 | 13821 | 3008 | 0 | 21.76% |
| round:round2 | 7508 | 1491 | 0 | 19.86% |

## 4. 各数据集 major 占比（降序）

| dataset | 单元数 | major | major 占比 |
|---|---:|---:|---:|
| Francesco__construction-safety-gsnvb | 5 | 0 | 0.00% |
| XimiaoZhang__MVTec-4K | 36 | 0 | 0.00% |
| ZhiyaYang__sewer-defect-crack-dataset | 3 | 0 | 0.00% |
| aswin00000__ConstructionSiteCleanedDataSet | 11339 | 3007 | 26.52% |
| chandrabhuma__multi_building_defect_vqa | 2417 | 1 | 0.04% |
| jhboyo__ppe-dataset | 3 | 0 | 0.00% |
| keremberke__construction-safety-object-detection | 17 | 0 | 0.00% |
| keremberke__satellite-building-segmentation | 1 | 0 | 0.00% |
| baizhanquan__FireDetectionDataset-flame-forest-flameye-wildfire | 2 | 0 | 0.00% |
| Voxel51__hard-hat-detection | 3 | 0 | 0.00% |
| fireviewer__fire-smoke-detection-corpus-v1 | 3 | 0 | 0.00% |
| hayden-yuma__roadwork | 7478 | 1490 | 19.93% |
| hf-vision__hardhat | 4 | 0 | 0.00% |
| iluvvatar__wood_surface_defects | 10 | 1 | 10.00% |
| keremberke__hard-hat-detection | 2 | 0 | 0.00% |
| kevincluo__structure_wildfire_damage_classification | 6 | 0 | 0.00% |

## 5. 已知错例是否被捕获（校准）

| 已知错例 | 类型 | 模型判定 | 引用证据 |
|---|---|---|---|
| test-00000#37 | Eleven->七人(num) | major | num:Eleven people->七人; term:distribution kiosk->Kiosk |
| test-00001#248 | Eleven->七名(num) | major | num:Eleven workers->七名工人 |
| test-00000#333 | excavator->木工钻床(term) | major | term:excavator->木工钻床; term:wood->木工 |
| test-00001#793 | rear view 错译 | major | neg:The rear view of an excavator. Another e->一台挖掘机的位于另一台挖掘机的后方。 |
| train-00000#562 | cranes->软管(term) | major | term:two cranes with extended booms->两根带吊索的软管 |
| train-00001#221 | earth->地球(term) | major | term:lucus->露; term:towers->高楼 |
| test-00000#326 | 漏译 without PPE | major | term:Four excavators->四名挖掘机手; neg:another worker without PPE->另一名头戴白色安全帽的工人; omit:There is |
| train-00002#502 | dredging->排水(term) | ok |  |
| test-00001#601 | next to->运往(extra) | major | extra:Some more sandbags are next to a pile of->更多的沙袋正被运往背景中的一堆泥土。 |
| test-00002#521 | 荒芜凉(garble) | major | garble:relatively barren->相对荒芜凉; garble:overcast sky->阴沉郁的天空 |
| test-00002#982 | 的物/桶(garble) | major | term:with a bucket full of dirt->旁边放着一桶装满泥土的桶; garble:one worker in a red hard hat and dar |

## 6. major 样例（随机 25 条）

- **ConstructionSiteCleanedDataSet#693 · image_caption**
  - EN: The image shows a multi-level structure with scaffolding, with a worker in a yellow hard hat visible on one of the upper levels. A tower crane is positioned to 
  - ZH: 图像显示一个多层结构，带有脚手架，一名头戴黄色安全帽的工人位于其中一个上层。一台塔式起重机位于结构的右侧，向上并超出框架。地面上，各种建筑材料，包括成堆的钢筋和木托盘，散落在场地周围。
  - 判定: term wooden planks→木托盘
- **ConstructionSiteCleanedDataSet#425 · image_caption**
  - EN: In the foreground, a worker wearing a yellow hard hat is crouching and tying rebar at the edge of a formwork. Three workers with yellow hard hats are standing a
  - ZH: 前景中，一名头戴黄色安全帽的工人正对着脚手架管边缘处哭泣，并系着脚手架。三名头戴黄色安全帽的工人正站在脚手架框架上作业，其中一人弯身越过脚手架。画面右侧前景处可见一丛竹子，中景处可见多根金属管。背景中，一些钢筋可被放置在混凝土墙前。
  - 判定: term crouching and tying rebar at the edge of→正对着脚手架管边缘处哭泣，并系着脚手架; term rebar framework→脚手架框架; term put against the concrete wall→可被放置在混凝土墙前
- **roadwork#321 · scene_description**
  - EN: Road partially blocked by arrow board and drums. Work vehicle next to arrow board. Line of drums next to work vehicle. Work vehicle behind arrow board in work z
  - ZH: 道路被箭头指示板和隔离桶部分阻断。作业车辆驶向箭头指示板。隔离桶驶向作业车辆。作业车辆在箭头指示板后的方的作业区内。
  - 判定: term next to arrow board→驶向箭头指示板; term next to work vehicle→驶向作业车辆
- **ConstructionSiteCleanedDataSet#206 · image_caption**
  - EN: Five workers with yellow or red hard hats are working on an unfinished tunnel structure. On the left, on the top of the finished section, there is trolley.
  - ZH: 五名头戴黄色或红色安全帽的工人正在一条已完工隧道的结构内作业。左侧，在完工段上方，有一辆手推车。
  - 判定: neg unfinished tunnel structure→已完工隧道的结构; term finished section→完工段
- **ConstructionSiteCleanedDataSet#702 · rule_1_violation.reason**
  - EN: The person in black in the foreground is not wearing a hard hat and is wearing shorts. Some workers are not wearing hard hats on top of the embankment on the le
  - ZH: 前景中的黑衣人没有戴硬帽，而是穿着短裤。一些工人没有在左侧的嵌入物上戴硬帽。
  - 判定: term embankment→嵌入物
- **roadwork#122 · scene_description**
  - EN: Drums on far left side of road. Line of cones on left side of road. Cones next to work vehicles on right side of road.
  - ZH: 道路左侧设置隔离桶。道路左侧设置交通锥。交通锥接下来设置作业车辆道路右侧。
  - 判定: term Drums→隔离桶; garble Cones next to work vehicles on right sid→交通锥接下来设置作业车辆道路右侧。
- **ConstructionSiteCleanedDataSet#241 · image_caption**
  - EN: The image shows a large construction site with three workers wearing orange coveralls and yellow hard hats walking on the roof with rebar mesh on top of it. In 
  - ZH: 图片显示一个大型建筑工地，三名工人头戴橙色安全帽、身穿黄色安全帽在屋顶行走，屋顶上装有钢筋网。前景处有一个混凝土平台。背景中有一大片大型开挖区域，两侧均设有阶梯式混凝土挡土墙，两侧均设有钢筋侧墙。左侧有一道由绿色织物制成的、位于斜坡顶部的上方的围栏。右侧可见两排钢板桩。该场地四周被树木环绕。
  - 判定: term wearing orange coveralls and yellow hard→头戴橙色安全帽、身穿黄色安全帽
- **ConstructionSiteCleanedDataSet#959 · image_caption**
  - EN: The image shows a construction site with one worker handling or inspecting a bundle of vertical rebars. There are multiple bundles of horizontal rebars laid out
  - ZH: 图中显示一个建筑工地，一名工人正在处理或检查一叠垂直钢筋。地面上有多层水平钢筋向左铺设，背景中还有额外的垂直钢筋，表明与混凝土结构相关的的工作正在进行持续进行。
  - 判定: omit formworks and additional vertical rebars→背景中还有额外的垂直钢筋; garble indicating ongoing construction work rel→表明与混凝土结构相关的的工作正在进行持续进行
- **ConstructionSiteCleanedDataSet#414 · image_caption**
  - EN: Six workers are on top of the concrete slab of an unfinished structure. Two workers on the right are on the edge of the slab. The structure has a rebar side wal
  - ZH: 六名工人位于一块已完成结构的混凝土板的顶部。右侧两名工人位于该板边缘。该结构中存在一面可见钢筋侧墙。图像左下方，左侧堤岸上有一堆金属管。右侧堤岸上，紧邻一面由蓝色金属板制成的墙壁，有两堆板材。背景中有两只狮子。
  - 判定: neg unfinished structure→已完成结构; extra Two pavilions are in the background.→背景中有两只狮子。
- **roadwork#49 · scene_description**
  - EN: Cones on left sidewalk. TTC signs on right sidewalk. Cones next to parked cars on right side of road.
  - ZH: 左侧人行道上的有交通锥。右侧人行道上有临时交通控制标志。交通锥位于道路右侧停放的车辆前方。
  - 判定: term TTC signs→临时交通控制标志; term next to→前方
- **ConstructionSiteCleanedDataSet#282 · rule_1_violation.reason**
  - EN: The worker squatting in the middle is wearing a straw hat but not a hard hat.
  - ZH: 蹲在中间的工人戴着一顶草帽，但并非一顶硬草帽。
  - 判定: term hard hat→硬草帽
- **ConstructionSiteCleanedDataSet#938 · image_caption**
  - EN: Twelve workers are working on a tunnel structure. Five workers are standing on top of the finished concrete section of the tunnel. The remaining workers are on 
  - ZH: 十名工人正在隧道结构内作业。五名工人站在已完工混凝土段顶部。其余工人位于隧道入口前方的钢筋网架上。除一名工人外，所有工人都戴着黄色安全帽和红色工作服。左侧有一排板材堆。
  - 判定: num Twelve workers→十名工人
- **ConstructionSiteCleanedDataSet#66 · image_caption**
  - EN: The image shows a construction site with muddy terrain and standing water. A tunnel-like structure with a square opening is placed in the excavation trench. In 
  - ZH: 图像显示一个施工现场，地形泥泞湿，有积水。一个类似隧道的方形开口放置在挖掘沟中。在中心，有一堆金属板、一堆砖块和一摞木托盘，位于隧道开口的前部。在左侧，有两名工人可见于挖掘沟顶部，其中一人戴着一顶红色安全帽。在右侧，有另一名工人，戴着红色安全帽。一根粗大的黄色软管横穿前景。
  - 判定: term a stack of wooden planks→一摞木托盘
- **roadwork#107 · scene_description**
  - EN: Cones on edge of left sidewalk.
  - ZH: 左侧人行道上的有坑洼。
  - 判定: term Cones→坑洼
- **roadwork#210 · scene_description**
  - EN: Line of cones on right side of road next to work vehicle. Worker on right intersecting road.
  - ZH: 右侧路交通锥后接作业车辆。右侧工人相交道路。
  - 判定: term Line of cones on right side of road next→右侧路交通锥后接作业车辆; garble Worker on right intersecting road→右侧工人相交道路
- **ConstructionSiteCleanedDataSet#640 · image_caption**
  - EN: Six workers are on the finished section of the tunnel structure. One worker is sitting in front of the entrance of the unfinished section of the tunnel. The unf
  - ZH: 六名工人位于隧道已完工区段。一名工人坐在隧道已完工区段入口前方。已完工区段由金属管支撑。一名工人位于隧道与钢板桩之间的右侧空隙处。一名工人在左侧人行天桥上行走。隧道顶部堆放着钢筋和金属管等建筑材料。
  - 判定: neg The unfinished section is supported by m→已完工区段由金属管支撑。
- **ConstructionSiteCleanedDataSet#157 · image_caption**
  - EN: The image shows a construction site with three excavators operating in an area with mounds of dirt. In the background, there are unfinished buildings with color
  - ZH: 图中显示一个建筑工地，有三台挖掘机正在一片泥土堆旁作业。背景中可见已完工建筑物，带有彩色屋顶。远处背景中，树木之间可见一台塔式起重机。
  - 判定: neg unfinished buildings→已完工建筑物
- **ConstructionSiteCleanedDataSet#166 · image_caption**
  - EN: The image shows a visible excavator in the center, which is a KATO HD820 model. A tripod is next to the excavator. On the left, there is a drum roller. In the b
  - ZH: 画面中央可见一台挖掘机，该机型为卡特HD820型号。挖掘机前方有一台三脚架。左侧有一台鼓轮。背景中有一台轮式装载机。地面看起来平整，由裸露出的土构成。背景杂乱，远处可见建筑物和树木。
  - 判定: term KATO HD820→卡特HD820; neg uneven→平整; term hazy→杂乱
- **roadwork#417 · scene_description**
  - EN: Cone on corner of left sidewalk. Cones next to work vehicles on right intersecting road.
  - ZH: 左侧人行道拐角处的有交通锥。接下来是右侧路口处的有作业车辆。
  - 判定: term Cones next to work vehicles on right int→接下来是右侧路口处的有作业车辆
- **ConstructionSiteCleanedDataSet#427 · image_caption**
  - EN: An excavator and a concrete truck are next to each other in an excavation pit.
  - ZH: 一台挖掘机和一辆混凝土卡车正在一个基坑内相互靠近。
  - 判定: term next to each other→相互靠近
- **ConstructionSiteCleanedDataSet#818 · image_caption**
  - EN: The image shows a construction site with a large number of rebars laid out in a grid pattern. There are five workers visible, all wearing hard hats. A worker wi
  - ZH: 图像显示一个建筑工地，大量钢筋以网格状排列。可见五名工人，均佩戴安全帽。一名佩戴黄色安全帽的工人站在中心位置。另外四名工人位于由黄色金属杆和绿色安全网构成的围栏的前。其中三人正在搬运并捆绑钢筋。网格中存在黑色板材，由钢筋构成。多束钢筋散落在工地各处。右侧背景中有蓝色和白色建筑，左侧两侧均有带树木的道路。
  - 判定: term tying rebars→搬运并捆绑钢筋
- **ConstructionSiteCleanedDataSet#518 · image_caption**
  - EN: The image shows a construction site with multiple workers and materials. There are at least seven workers visible, wearing hard hats and engaged in various acti
  - ZH: 图像显示一个施工现场，有多名工人和材料。可见至少七名工人，头戴安全帽，正从事各种活动，例如搬运材料。现场堆满了建筑材料，包括成堆的木方材、成束的钢筋以及成叠的镀锌板。脚手架沿一栋在建建筑的的一侧搭设，背景中表明该建筑内部正在施工。
  - 判定: term indicating work on the building's exteri→表明该建筑内部正在施工
- **ConstructionSiteCleanedDataSet#462 · image_caption**
  - EN: An excavator is in the middle. Eight concrete pipe sections are on two sides of the road with four sections on each side.
  - ZH: 一台挖掘机位于道路中央。道路两侧各有八节混凝土管段。
  - 判定: num Eight concrete pipe sections are on two →道路两侧各有八节混凝土管段
- **roadwork#392 · scene_description**
  - EN: Cones on left side of road.
  - ZH: 道路左侧的坑洼。
  - 判定: term Cones→坑洼
- **ConstructionSiteCleanedDataSet#235 · image_caption**
  - EN: Four workers are on top of the tunnel structure under construction. The worker on the right with a camouflage jacket is holding a rebar in his hand. One stack o
  - ZH: 四名工人位于隧道结构下方施工处。右侧那名工人身穿迷彩服，手中握着一根钢筋。一根钢筋紧挨着隧道。背景中有一条河流，以及一名头戴黄色安全帽的工人紧挨着河流。
  - 判定: neg on top of the tunnel structure→隧道结构下方; num One stack of rebar→一根钢筋

*失败条目：15；完整清单见 `flagged_major.jsonl`*
