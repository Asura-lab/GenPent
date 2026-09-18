# GenPent — Generalizable Autonomous Penetration Testing with RL

Reinforcement Learning ашиглан автомат penetration testing агент сургах, түүний **үзээгүй сүлжээний топологи руу generalize хийх чадварыг** судлах төсөл.

## Гол судалгааны асуулт

> NASim симуляторт сурсан pentesting агент үзээгүй сүлжээний топологи (өөр хэмжээ, өөр хөрш зээл, өөр эмзэг байдал) руу хэрхэн шилжих вэ?

## Хар алсын хараа (roadmap)

| Фаз | Хугацаа | Ажил | Статус |
|-----|---------|------|--------|
| 0 | 1 долоо хоног | Төслийн суурь, NASim wrapper, baseline агентууд | 🚧 |
| 1 | 3–4 долоо хоног | DQN/PPO deep baselines, multi-seed үнэлгээ | ⏳ |
| 2 | 4–6 долоо хоног | Recurrent policy, action masking, curiosity (ICM/RND) | ⏳ |
| 3 | 4–6 долоо хоног | Generalization harness — generated топологи дээр train→test | ⏳ |
| 4 | сонголтоор | Stealth-aware reward, LLM hybrid, NASimEmu sim-to-real | ⏳ |

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
