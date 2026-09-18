# GenPent — Generalizable Autonomous Penetration Testing with RL

Reinforcement Learning ашиглан автомат penetration testing агент сургах, түүний **үзээгүй сүлжээний топологи руу generalize хийх чадварыг** судлах төсөл.

## Гол судалгааны асуулт

> NASim симуляторт сурсан pentesting агент үзээгүй сүлжээний топологи (өөр хэмжээ, өөр хөрш зээл, өөр эмзэг байдал) руу хэрхэн шилжих вэ?

## Хар алсын хараа (roadmap)

| Фаз | Хугацаа | Ажил | Статус |
|-----|---------|------|--------|
| 0 | 1 долоо хоног | Төслийн суурь, NASim wrapper, baseline агентууд | ✅ |
| 1 | 3–4 долоо хоног | DQN/MaskablePPO baselines, multi-seed үнэлгээ | ✅ |
| 2 | 4–6 долоо хоног | Recurrent policy, reward shaping, curiosity (ICM/RND) | ⏳ |
| 3 | 4–6 долоо хоног | Generalization harness — generated топологи дээр train→test | 🚧 Эхний үр дүн гарлаа |
| 4 | сонголтоор | Stealth-aware reward, LLM hybrid, NASimEmu sim-to-real | ⏳ |

## Одоогийн үр дүн (2026-09-18)

**Tiny scenario (50 episode, GPU):**

| Агент | Success | Mean reward | Steps (median) |
|-------|---------|-------------|----------------|
| random_masked | 100% | 81.9 | 114.5 |
| bruteforce | 100% | 100.1 | 94.5 |
| **MaskPPO (3 seed)** | **100%** | **192–193** | **6–7** |

**Generalization (10 hosts, үзээгүй 5 топологи):**

| Сургалт | Сургалтын алхам | Амжилттай топологи |
|---------|------------------|---------------------|
| Ensemble, shaping=0 | 200k | 1/5 |
| **Ensemble, shaping=100** | **300k** | **2/5** |
| Single, shaping=0 | 200k | 0/5 |
| Single, shaping=100 | 300k | 0/5 |

**Гол олдворууд:**
1. DQN нь invalid action дээр Q-overestimation loop-д ордог — action mask
   (MaskablePPO) нь шийдэл: 3/3 seed 100% success
2. NASim-ийн sparse reward + ε-greedy нь seed-sensitive — DQN 1/3 seed л сурсан
3. **Potential-based reward shaping** (Φ = discovery+access+root) нь small
   scenario-г 0% → 100% болгосон (400k алхам, 280k дээр "grokking")
4. **Shaping нь generalization-д ч тусалсан**: ensemble 1/5 → 2/5,
   single хэзээ ч 0/5 — олон топологи + shaping хослол нь шилжилтийн урьд
   нөхцөл
5. Small scenario нь episode урт (1000 алхам) тул 150k алхамд ч суралцаагүй —
   reward shaping эсвэл curriculum хэрэгтэй (Фаз 2) — ✅ шийдэгдсэн

## MDP тодорхойлолт (NASim)

- **State:** сүлжээний бүрэн төлөв (host бүрийн discovered/compromised/reachable, access level)
- **Observation:** хэсэгчилсэн — зөвхөн нээгдсэн host-ийн мэдээлэл (POMDP)
- **Action:** scan / exploit / privilege escalation / noop (flat action space, NASim 0.12)
- **Reward:** exploit амжилттай бол host-ийн утга, discovery-д бага утга, алхам бүрт −1
- **Episode:** goal (sensitive host-уудад root access) эсвэл step_limit хүртэл

## Хавтасны бүтэц

```
GenPent/
├── envs/
│   └── nasim_gym.py        - Gymnasium wrapper (obs normalization, action mask)
├── agents/
│   ├── random_agent.py     - Random baseline
│   └── bruteforce_agent.py - Бүх үйлдлийг ээлжлэн турших baseline
├── evaluation/
│   ├── evaluate.py         - N episode-ийн дүн (success rate, steps, reward)
│   └── baselines.py        - Baseline агентуудыг ажиллуулж CSV хадгалах
├── tests/
├── configs/
│   └── eval_config.yaml    - Үнэлгээний тохиргоо (scenario, seed, episode тоо)
├── results/                - CSV/plot үр дүн (git-д орохгүй)
├── notebooks/
└── requirements.txt
```

## Суулгах

```bash
# Desktop/RL дотор аль хэдийн GenPent-venv үүссэн байгаа
cd Desktop/RL
python -m venv GenPent-venv          # анх удаа
GenPent-venv/Scripts/activate

pip install -r GenPent/requirements.txt
```

## GPU тохиргоо

Сургалт GPU дээр ажиллана (`device="cuda"`). CPU-г CUDA torch руу шилжүүлэх:

```bash
pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu130
```

Анхаарах зүйлс:
- NASim симуляци нь Python loop учраас CPU дээр ажилладаг — GPU нь нейрон
  сүлжээний forward/backward-т ашиглагдана
- Tiny/Small scenario-д сүлжээ жижиг тул GPU давуу тал бага — Medium/Large,
  том batch-т GPU жинхэнэ хурд өгнө
- Монгол лог хэвлэхэд `PYTHONIOENCODING=utf-8` хэрэгтэй (Windows)

## Ашиглах

**Baseline үнэлгээ ажиллуулах**
```bash
cd Desktop/RL/GenPent
../GenPent-venv/Scripts/python evaluation/baselines.py --scenarios tiny small --episodes 20
```

**Тест**
```bash
../GenPent-venv/Scripts/python -m pytest tests/ -v
```

## Өгөгдлийн аюулгүй байдлын дүрэм

- Энэ төсөл зөвхөн **симуляторт** ажилладаг — NASim нь виртуал сүлжээ, бодит системд халддаггүй
- Бодит сүлжээнд ашиглахыг зөвхөн зөвшөөрөлтэй pentest-ийн хүрээнд

## Лиценз

MIT License

## Ишлэх

Хэрэв энэ төслийг ашигласан бол NASim-ийг ишлэ:

```bibtex
@misc{schwartz2019nasim,
  title={NASim: Network Attack Simulator},
  author={Schwartz, Jonathon and Kurniawatti, Hanna},
  year={2019},
  howpublished={\url{https://networkattacksimulator.readthedocs.io/}},
}
```
