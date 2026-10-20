# Чужие решения и литература: что взято и что проверено

Состояние на 21.10.2026. Код и веса других команд в решение не попадают (ТР: идентичные продукты запрещены, п. 13.4–13.5 — про лицензии). Из чужих материалов берутся только идеи. Каждую идею мы реализуем сами и проверяем официальной метрикой вне обучения, по «семьям» участков. Ссылки на людей не собираются, только на открытые репозитории и публикации.

## Что найдено

Поиск по GitHub: имена репозиториев (`expds`, `ds-expedition`, `koz-3`, `expds-solution`), код (`kurgany_tselye`, `kurgany_povrezhdennye`, `finds_points`, `train-koz-3`, `markup_type etgr`) и описания. Других репозиториев КОЗ №3 с баллами на настоящих данных не нашлось.

| Источник | Что это | Чем подтверждено |
|---|---|---|
| [yakirill99/ds-expedition](https://github.com/yakirill99/ds-expedition) | Команда КОЗ №3, каркас. U-Net с двумя головами: маска и гауссова карта центров (как CenterNet); 18 каналов рельефа: SLRM, положительная и отрицательная открытость, SVF, кривизна, TPI, TRI, четыре отмывки, плотность и интенсивность точек; фильтр земли PMF с «мягким» режимом для насыпей; склейка окном Ханна | Только синтетика: F1 0,71. Из реальных данных обработан один участок с лидаром, без метрики |
| [KateBuikina/expds-solution](https://github.com/KateBuikina/expds-solution) | Курсовая на данных КОЗ №2: U-Net ResNet34 на каждую модальность, классификатор модальности (MobileNetV3), focal + Dice, температурная калибровка (T = 0,85), mosaic, сглаживание меток | Баллов в репозитории нет |
| [MataNerdy/Geodata_Archaeology_CV](https://github.com/MataNerdy/Geodata_Archaeology_CV) | Исследование на данных серии: U-Net, DeepLabV3+, YOLO | Взвешенный F1 0,7457 у DeepLabV3+ на своём разбиении; вывод: постобработка меняет порядок моделей, нужна проверка по регионам |
| [Egor-git-cloud/ds-expedition](https://github.com/Egor-git-cloud/ds-expedition) | Каркас и адаптер бейзлайна организаторов | Обучения нет |
| [L0thlorien/ds-expedition](https://github.com/L0thlorien/ds-expedition), [KoshelevDV/ds-expedition-infra](https://github.com/KoshelevDV/ds-expedition-infra), [iwannasome/expds-rubleincorp-KOZ2](https://github.com/iwannasome/expds-rubleincorp-KOZ2) | EDA, инфраструктура, пустой репозиторий (КОЗ №2) | — |
| Описания ПО призёров КОЗ №1–2 на сайте Фонда НТИ (`files/sources/nti`: «Северная звезда», Family, Orbis, GigaFlex, mlart) | Общее: ансамбли (Family КОЗ №2 — восемь весов, Orbis — состав и веса в `ensemble_config.yaml`, mlart — ансамбль YOLO), пороги по классам, минимальные площади, тайлы с перекрытием, удаление дублей | Призовые места в прошлых конкурсах серии |

## Идеи и решения по ним

| Идея | Где встречается | У нас | Итог |
|---|---|---|---|
| Ансамбль моделей | Family, Orbis, mlart; глубокие ансамбли снижают излишнюю уверенность вне обучающих данных [1, 2] | `infer_nn.Ensemble`, `[models] li = {path = [...]}`, `nn_oof.py --seeds` | Проверяется: две модели на фолд с разными зёрнами, те же фолды, что у `li_r18_hard2.pt` |
| Положительная открытость и SVF как каналы | yakirill99; в сравнениях визуализаций для сверточных сетей они в верхней группе, локальный рельеф слабее [5, 6] | `opns`, `svf` в `lidar.relief_channels`, Rust `openness_svf` (16 лучей по 10 м) | Проверяется на том же кэше с пятью каналами. На центрах курганов открытость ниже фона (соседние насыпи поднимают горизонт), поэтому без проверки вне обучения выводов нет |
| Пороги по классам и минимальные площади | все призёры | Подбор порогов официальной метрикой за один проход; пределы площадей геофизики по описанию классов | Уже было |
| Проверка по регионам; постобработка меняет порядок моделей | MataNerdy | Проверка только по «семьям» участков, все решения — по OOF | Уже было |
| Simple Copy-Paste [7] | — | `train --copy-paste`, только на пустые окна | Хуже: 0,288 против 0,310 на 49 участках с лидаром (вместе с новыми LAS Гнёздова). Выключено |
| Подстройка BatchNorm под снимок (AdaBN) [8] | — | Проверено на двух участках | Ложных пиков на пустом участке 3208 вместо 13. Не вошло |
| Карта центров (CenterNet) [9] | yakirill99 | Центры даёт пиковый поиск по рельефу, U-Net их перевзвешивает | Не делали: у автора нет сравнения на реальных данных, наш путь проверен вне обучения |
| Температурная калибровка [10] | KateBuikina | — | Не нужна: монотонное преобразование уверенности не меняет F1 при пороге по классу |
| Склейка окном Ханна, субпиксельный пик | yakirill99 | Треугольное окно; круг 2 м | Не влияет на метрику: курган засчитывается по внутренней точке и покрытию |
| Фильтр земли PMF [11] | yakirill99 | Класс 2 из LAS | Не трогаем: карты Li организаторов, по которым размечали, строятся по классу 2 |
| Классификатор модальности | KateBuikina | По кодам и словам в именах файлов | Не нужен |
| TTA отражениями | — | Проверено | Без выигрыша, выключено |

## Найдено попутно

Базовые окна кэша, на котором учились текущие веса, расходятся с кодом инференса. На Тимереве различаются 1,4–1,6 % пикселей, местами до 157 уровней из 255. DEM при этом побитово тот же. Значит, окна собирала более старая версия каналов. Кэш `li_cur3_hard2_t128` пересчитан текущим кодом на тех же центрах и с теми же масками. Проверка вне обучения на нём покажет, сколько стоит это рассогласование.

## Литература

1. Lakshminarayanan B., Pritzel A., Blundell C. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017.
2. Ovadia Y. et al. Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. NeurIPS 2019.
3. Yokoyama R., Shirasawa M., Pike R. J. Visualizing topography by openness. Photogrammetric Engineering & Remote Sensing 68(3), 2002.
4. Zakšek K., Oštir K., Kokalj Ž. Sky-View Factor as a Relief Visualization Technique. Remote Sensing 3(2), 2011.
5. Guyot A., Lennon M., Hubert-Moy L. Objective comparison of relief visualization techniques with deep CNN for archaeology. Journal of Archaeological Science: Reports 38, 2021. [doi:10.1016/j.jasrep.2021.103027](https://doi.org/10.1016/j.jasrep.2021.103027)
6. Prospecting of Architectural Features Using LiDAR-UAV Technology, Deep Neural Networks and Visualization Techniques: A Case Study in Kuélap and Cambolín (NW Peru). Archaeological Prospection, 2025. [doi:10.1002/arp.70033](https://doi.org/10.1002/arp.70033) — на одном участке лучше VAT, открытость и SVF, на другом уклон.
7. Ghiasi G. et al. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. CVPR 2021.
8. Li Y. et al. Adaptive Batch Normalization for practical domain adaptation. Pattern Recognition 80, 2018.
9. Zhou X., Wang D., Krähenbühl P. Objects as Points. arXiv:1904.07850, 2019.
10. Guo C. et al. On Calibration of Modern Neural Networks. ICML 2017.
11. Zhang K. et al. A progressive morphological filter for removing nonground measurements from airborne LIDAR data. IEEE TGRS 41(4), 2003.
12. Hesse R. LiDAR-derived Local Relief Models — a new tool for archaeological prospection. Archaeological Prospection 17(2), 2010.
13. Perron Y. et al. Archaeoscape: Bringing Aerial Laser Scanning Archaeology to the Deep Learning Era. NeurIPS 2024 Datasets and Benchmarks, arXiv:2412.05203.
