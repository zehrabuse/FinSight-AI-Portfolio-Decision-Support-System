import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):

    def __init__(self, scenarios):

        super().__init__()

        self.scenarios = scenarios

        # ============================================================
        # SENARYO İSİMLERİ
        # ============================================================

        # Senaryo datasetlerinin kendisi public repository'de
        # paylaşılmamaktadır.
        #
        # Burada yalnızca environment mantığını ve senaryo
        # seçim yapısını göstermek amacıyla senaryo isimleri
        # korunmaktadır.

        self.scenario_names = {
            1: "Normal Piyasa",
            2: "COVID-19 Krizi",
            3: "2008 Benzeri Sert Çöküş",
            4: "Spekülatif Balon ve Ani Yükseliş",
            5: "Tarihsel ve Jeopolitik Krizler",
            6: "Belirsiz Yüksek Volatilite",
            7: "Sentetik Rastgele Krizler",
            8: "Flash Crash",
            9: "Aşırı Volatilite ve Overtrading",
            10: "Uzun Ayı Piyasası",
            11: "Uzun Boğa Piyasası",
            12: "Yatay Piyasa ve Overtrading",
            13: "Enflasyon Şoku",
            14: "Faiz Şoku",
            15: "Jeopolitik Kriz",
            16: "Black Swan / Kara Kuğu"
        }

        # ============================================================
        # SENARYO SEÇİMİ
        # ============================================================

        scenario_index = random.randrange(len(self.scenarios))

        self.current_scenario = scenario_index + 1
        self.data = self.scenarios[scenario_index]

        self.n_steps = len(self.data)
        self.current_step = 0

        # ============================================================
        # BAŞLANGIÇ PORTFÖYÜ
        # ============================================================

        self.initial_cash = 100000
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash

        # ============================================================
        # MOCK PORTFÖY AĞIRLIKLARI
        #
        # Bu değerler tamamen sentetik/mock değerlerdir.
        # Gerçek bir portföy dağılımını veya gerçek piyasa verisini
        # temsil etmez.
        #
        # Public repository'de environment mantığını göstermek
        # amacıyla kullanılmaktadır.
        # ============================================================

        self.initial_stock_weight = 0.85
        self.initial_repo_weight = 0.07
        self.initial_collateral_weight = 0.03
        self.initial_fund_weight = 0.05

        self.stock_weight = self.initial_stock_weight
        self.repo_weight = self.initial_repo_weight
        self.collateral_weight = self.initial_collateral_weight
        self.fund_weight = self.initial_fund_weight

        # ============================================================
        # MOCK HİSSE KIRILIMI
        #
        # Bu değerler tamamen sentetik/mock değerlerdir.
        # Gerçek portföy dağılımını veya gerçek piyasa verisini
        # temsil etmez.
        #
        # Public repository'de environment mantığını göstermek
        # amacıyla kullanılmaktadır.
        # ============================================================

        self.stock_breakdown = {
            "MOCK_STOCK_01": 0.11,
            "MOCK_STOCK_02": 0.10,
            "MOCK_STOCK_03": 0.09,
            "MOCK_STOCK_04": 0.08,
            "MOCK_STOCK_05": 0.07,
            "MOCK_STOCK_06": 0.06,
            "MOCK_STOCK_07": 0.05,
            "MOCK_STOCK_08": 0.04,
            "MOCK_STOCK_09": 0.03,
            "MOCK_STOCK_10": 0.02,
            "OTHER": 0.35
        }

        # Başlangıç hisse kırılımını koru.
        self.initial_stock_breakdown = self.stock_breakdown.copy()

        # ============================================================
        # HİSSE KIRILIMI DEĞİŞİM LİMİTİ
        # ============================================================

        # Her hisse bir step'te maksimum 5 yüzde puanı değişebilir.
        self.max_stock_breakdown_change = 0.05

        # ============================================================
        # REWARD İÇİN GEÇMİŞ PORTFÖY GETİRİLERİ
        # ============================================================

        self.return_history = []

        # ============================================================
        # TERMINATION PARAMETRELERİ
        # ============================================================

        # Portföy %45 kayıp yaşarsa episode sonlandırılır.
        self.min_portfolio_ratio = 0.55

        # Portföy başlangıç değerinin 2 katına ulaşırsa
        # episode sonlandırılır.
        self.target_portfolio_ratio = 2.0

        # ============================================================
        # ACTION CONSTRAINTS
        # ============================================================

        # Tek işlemde maksimum portföy ağırlığı değişimi: %10
        self.max_weight_change = 0.10

        # Hisse ağırlığı minimum %80 olmalıdır.
        self.min_stock_weight = 0.80

        # ============================================================
        # ACTION SPACE
        # ============================================================

        self.action_space = spaces.Discrete(9)

        # ============================================================
        # STATE SPACE
        # ============================================================

        # DQN modeli 16 boyutlu state almaktadır.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32
        )

        # ============================================================
        # ACTION İSİMLERİ
        # ============================================================

        self.action_names = {
            0: "HOLD",
            1: "BUY_STOCK",
            2: "SELL_STOCK",
            3: "BUY_REPO",
            4: "SELL_REPO",
            5: "BUY_COLLATERAL",
            6: "SELL_COLLATERAL",
            7: "BUY_FUND",
            8: "SELL_FUND"
        }

    # ================================================================
    # RESET
    # ================================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Her episode için rastgele senaryo seç
        scenario_index = random.randrange(len(self.scenarios))

        self.current_scenario = scenario_index + 1
        self.data = self.scenarios[scenario_index]

        self.n_steps = len(self.data)
        self.current_step = 0

        # ============================================================
        # PORTFÖYÜ BAŞLANGIÇ DURUMUNA GETİR
        # ============================================================

        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash

        # ============================================================
        # RİSK / RETURN GEÇMİŞİNİ SIFIRLA
        # ============================================================

        self.return_history = []

        # ============================================================
        # BAŞLANGIÇ PORTFÖY AĞIRLIKLARINI SIFIRLA
        # ============================================================

        self.stock_weight = self.initial_stock_weight
        self.repo_weight = self.initial_repo_weight
        self.collateral_weight = self.initial_collateral_weight
        self.fund_weight = self.initial_fund_weight

        # ============================================================
        # HİSSE KIRILIMINI BAŞLANGIÇ DURUMUNA GETİR
        # ============================================================

        self.stock_breakdown = self.initial_stock_breakdown.copy()

        # Ağırlıkların toplamı %100 değilse,
        # eksik kısmı hisseye ekle.

        total_weight = (
            self.stock_weight
            + self.repo_weight
            + self.collateral_weight
            + self.fund_weight
        )

        if total_weight < 1.0:
            self.stock_weight += (1.0 - total_weight)

        state = self._get_state()

        info = {}

        return state, info

    # ================================================================
    # STATE
    # ================================================================

    def _get_state(self):

        row = self.data.iloc[self.current_step]

        cds = row["CDS_Spread_bps"] / 1000
        inflation = row["Annual_Inflation"] / 100
        policy_rate = row["Policy_Rate"] / 100

        fund_return = row["Fund_Return"]
        portfolio_growth = row["Portfolio_Growth"]

        active_value = row["Active_Value"] / 1e9
        cash_value = row["Cash_Value"] / 1e6
        investor_count = row["Investor_Count"] / 100000

        state = np.array(

            [
                # ----------------------------------------------------
                # Makroekonomik Özellikler
                # ----------------------------------------------------

                row["USD_Return"],
                row["Gold_Return"],
                row["Brent_Return"],
                row["US10Y_Return"],
                cds,
                inflation,
                policy_rate,

                # ----------------------------------------------------
                # Fon ve Portföy Verileri
                # ----------------------------------------------------

                fund_return,
                portfolio_growth,
                active_value,
                cash_value,
                investor_count,

                # ----------------------------------------------------
                # Ajanın Portföy Ağırlıkları
                # ----------------------------------------------------

                self.stock_weight,
                self.repo_weight,
                self.collateral_weight,
                self.fund_weight
            ],

            dtype=np.float32
        )

        return state

    # ================================================================
    # STEP
    # ================================================================

    def step(self, action):

        today = self.data.iloc[self.current_step]

        # ============================================================
        # ACTION VARIABLES
        # ============================================================

        trade_amount = 0.0
        trade_source = None
        trade_target = None
        full_liquidation = False
        available = 0.0

        # ============================================================
        # BUY STOCK
        # ============================================================

        if action == 1:

            remaining = self.max_weight_change

            old_stock_weight = self.stock_weight

            # --------------------------------------------------------
            # REPO -> STOCK
            # --------------------------------------------------------

            transfer = min(
                self.repo_weight,
                remaining
            )

            self.repo_weight -= transfer
            self.stock_weight += transfer
            remaining -= transfer

            if transfer > 0:
                trade_source = "TPP"
                trade_target = "HS"

            # --------------------------------------------------------
            # COLLATERAL -> STOCK
            # --------------------------------------------------------

            if remaining > 0:

                transfer = min(
                    self.collateral_weight,
                    remaining
                )

                self.collateral_weight -= transfer
                self.stock_weight += transfer
                remaining -= transfer

            # --------------------------------------------------------
            # FUND -> STOCK
            # --------------------------------------------------------

            if remaining > 0:

                transfer = min(
                    self.fund_weight,
                    remaining
                )

                self.fund_weight -= transfer
                self.stock_weight += transfer
                remaining -= transfer

            trade_amount = self.max_weight_change - remaining

            # ========================================================
            # HİSSE KIRILIMINI GÜNCELLE
            # ========================================================

            if trade_amount > 0:

                old_breakdown = self.stock_breakdown.copy()

                base_weights = self.initial_stock_breakdown

                adjusted_weights = {}

                for stock, weight in base_weights.items():
                    adjusted_weights[stock] = weight ** 1.05

                total_adjusted = sum(
                    adjusted_weights.values()
                )

                for stock in adjusted_weights:
                    adjusted_weights[stock] /= total_adjusted

                proposed_breakdown = {}

                if self.stock_weight > 0:

                    for stock in self.stock_breakdown:

                        old_amount = (
                            old_stock_weight
                            * old_breakdown[stock]
                        )

                        new_amount = (
                            trade_amount
                            * adjusted_weights[stock]
                        )

                        total_amount = (
                            old_amount + new_amount
                        )

                        proposed_breakdown[stock] = (
                            total_amount / self.stock_weight
                        )

                # ----------------------------------------------------
                # HER HİSSE İÇİN ±5 YÜZDE PUANI KISITI
                # ----------------------------------------------------

                max_change = self.max_stock_breakdown_change

                lower_bounds = {}
                upper_bounds = {}

                for stock in old_breakdown:

                    lower_bounds[stock] = max(
                        0.0,
                        old_breakdown[stock] - max_change
                    )

                    upper_bounds[stock] = min(
                        1.0,
                        old_breakdown[stock] + max_change
                    )

                # ----------------------------------------------------
                # BOUNDED PROJECTION
                # ----------------------------------------------------

                stocks = list(proposed_breakdown.keys())

                lower_sum = sum(
                    lower_bounds.values()
                )

                upper_sum = sum(
                    upper_bounds.values()
                )

                if lower_sum <= 1.0 <= upper_sum:

                    low = min(
                        proposed_breakdown[s]
                        - upper_bounds[s]
                        for s in stocks
                    )

                    high = max(
                        proposed_breakdown[s]
                        - lower_bounds[s]
                        for s in stocks
                    )

                    for _ in range(100):

                        mid = (low + high) / 2

                        total = 0.0

                        for stock in stocks:

                            value = np.clip(
                                proposed_breakdown[stock] - mid,
                                lower_bounds[stock],
                                upper_bounds[stock]
                            )

                            total += value

                        if total > 1.0:
                            low = mid
                        else:
                            high = mid

                    for stock in stocks:

                        self.stock_breakdown[stock] = np.clip(
                            proposed_breakdown[stock] - high,
                            lower_bounds[stock],
                            upper_bounds[stock]
                        )

                else:

                    self.stock_breakdown = old_breakdown.copy()

            # --------------------------------------------------------
            # Son güvenlik: toplam %100
            # --------------------------------------------------------

            total = sum(
                self.stock_breakdown.values()
            )

            if total > 0:

                for stock in self.stock_breakdown:
                    self.stock_breakdown[stock] /= total

        # ============================================================
        # SELL STOCK
        # ============================================================

        elif action == 2:

            available = max(
                0.0,
                self.stock_weight - self.min_stock_weight
            )

            trade_amount = min(
                available,
                self.max_weight_change
            )

            if trade_amount > 0:

                old_stock_weight = self.stock_weight

                self.stock_weight -= trade_amount
                self.repo_weight += trade_amount

                trade_source = "HS"
                trade_target = "TPP"

                # ----------------------------------------------------
                # HİSSELERİ ORANSAL OLARAK SAT
                # ----------------------------------------------------

                if old_stock_weight > 0:

                    remaining_ratio = (
                        self.stock_weight
                        / old_stock_weight
                    )

                    for stock in self.stock_breakdown:

                        self.stock_breakdown[stock] *= (
                            remaining_ratio
                        )

                # ----------------------------------------------------
                # Normalize
                # ----------------------------------------------------

                total = sum(
                    self.stock_breakdown.values()
                )

                if total > 0:

                    for stock in self.stock_breakdown:
                        self.stock_breakdown[stock] /= total

        # ============================================================
        # BUY REPO
        # ============================================================

        elif action == 3:

            remaining = self.max_weight_change

            # --------------------------------------------------------
            # STOCK -> REPO
            # --------------------------------------------------------

            available = max(
                0.0,
                self.stock_weight - self.min_stock_weight
            )

            transfer = min(
                available,
                remaining
            )

            self.stock_weight -= transfer
            self.repo_weight += transfer
            remaining -= transfer

            if transfer > 0:
                trade_source = "HS"
                trade_target = "TPP"

            # --------------------------------------------------------
            # COLLATERAL -> REPO
            # --------------------------------------------------------

            if remaining > 0:

                transfer = min(
                    self.collateral_weight,
                    remaining
                )

                self.collateral_weight -= transfer
                self.repo_weight += transfer
                remaining -= transfer

            # --------------------------------------------------------
            # FUND -> REPO
            # --------------------------------------------------------

            if remaining > 0:

                transfer = min(
                    self.fund_weight,
                    remaining
                )

                self.fund_weight -= transfer
                self.repo_weight += transfer
                remaining -= transfer

            trade_amount = (
                self.max_weight_change - remaining
            )

        # ============================================================
        # SELL REPO
        # ============================================================

        elif action == 4:

            available = self.repo_weight

            trade_amount = min(
                available,
                self.max_weight_change
            )

            if trade_amount > 0:

                self.repo_weight -= trade_amount
                self.stock_weight += trade_amount

                trade_source = "TPP"
                trade_target = "HS"

        # ============================================================
        # BUY COLLATERAL
        # ============================================================

        elif action == 5:

            remaining = self.max_weight_change

            # --------------------------------------------------------
            # STOCK -> COLLATERAL
            # --------------------------------------------------------

            available = max(
                0.0,
                self.stock_weight - self.min_stock_weight
            )

            transfer = min(
                available,
                remaining
            )

            self.stock_weight -= transfer
            self.collateral_weight += transfer
            remaining -= transfer

            if transfer > 0:
                trade_source = "HS"
                trade_target = "T"

            # --------------------------------------------------------
            # REPO -> COLLATERAL
            # --------------------------------------------------------

            if remaining > 0:

                transfer = min(
                    self.repo_weight,
                    remaining
                )

                self.repo_weight -= transfer
                self.collateral_weight += transfer
                remaining -= transfer

            # --------------------------------------------------------
            # FUND -> COLLATERAL
            # --------------------------------------------------------

            if remaining > 0:

                transfer = min(
                    self.fund_weight,
                    remaining
                )

                self.fund_weight -= transfer
                self.collateral_weight += transfer
                remaining -= transfer

            trade_amount = (
                self.max_weight_change - remaining
            )

        # ============================================================
        # SELL COLLATERAL
        # ============================================================

        elif action == 6:

            available = self.collateral_weight

            trade_amount = min(
                available,
                self.max_weight_change
            )

            if trade_amount > 0:

                self.collateral_weight -= trade_amount
                self.stock_weight += trade_amount

                trade_source = "T"
                trade_target = "HS"

        # ============================================================
        # BUY FUND
        # ============================================================

        elif action == 7:

            remaining = self.max_weight_change

            # --------------------------------------------------------
            # STOCK -> FUND
            # --------------------------------------------------------

            available = max(
                0.0,
                self.stock_weight - self.min_stock_weight
            )

            transfer = min(
                available,
                remaining
            )

            self.stock_weight -= transfer
            self.fund_weight += transfer
            remaining -= transfer

            if transfer > 0:
                trade_source = "HS"
                trade_target = "YYF"

            # --------------------------------------------------------
            # REPO -> FUND
            # --------------------------------------------------------

            if remaining > 0:

                transfer = min(
                    self.repo_weight,
                    remaining
                )

                self.repo_weight -= transfer
                self.fund_weight += transfer
                remaining -= transfer

            # --------------------------------------------------------
            # COLLATERAL -> FUND
            # --------------------------------------------------------

            if remaining > 0:

                transfer = min(
                    self.collateral_weight,
                    remaining
                )

                self.collateral_weight -= transfer
                self.fund_weight += transfer
                remaining -= transfer

            trade_amount = (
                self.max_weight_change - remaining
            )

        # ============================================================
        # SELL FUND
        # ============================================================

        elif action == 8:

            available = self.fund_weight

            trade_amount = min(
                available,
                self.max_weight_change
            )

            if trade_amount > 0:

                self.fund_weight -= trade_amount
                self.stock_weight += trade_amount

                trade_source = "YYF"
                trade_target = "HS"

        # ============================================================
        # HARD CONSTRAINTS
        # ============================================================

        self.stock_weight = np.clip(
            self.stock_weight,
            self.min_stock_weight,
            1.00
        )

        self.repo_weight = np.clip(
            self.repo_weight,
            0.00,
            0.20
        )

        self.collateral_weight = np.clip(
            self.collateral_weight,
            0.00,
            0.20
        )

        self.fund_weight = np.clip(
            self.fund_weight,
            0.00,
            0.20
        )

        # ============================================================
        # GÜNLÜK GETİRİ
        # ============================================================

        fund_return = today["Fund_Return"]

        if pd.isna(fund_return):
            fund_return = 0.0

        # Senaryo verisindeki hedef portföy ağırlıkları ile
        # mevcut portföy ağırlıkları arasındaki fark.

        deviation = (
            abs(self.stock_weight - today["HS"])
            + abs(self.repo_weight - today["TPP"])
            + abs(self.collateral_weight - today["T"])
            + abs(self.fund_weight - today["YYF"])
        )

        # Sapmaya göre getiriyi ölçekleyen katsayı
        portfolio_multiplier = max(
            0.8,
            1 - deviation
        )

        daily_return = (
            fund_return * portfolio_multiplier
        )

        # ============================================================
        # PORTFÖY GÜNCELLEME
        # ============================================================

        previous_value = self.portfolio_value

        self.portfolio_value *= (
            1 + daily_return
        )

        # ============================================================
        # PORTFÖY GETİRİSİ
        # ============================================================

        portfolio_return = (
            self.portfolio_value - previous_value
        ) / previous_value

        # ============================================================
        # İŞLEM MALİYETİ
        # ============================================================

        transaction_cost = -0.01 * trade_amount

        # ============================================================
        # GEÇMİŞ GETİRİYİ KAYDET
        # ============================================================

        self.return_history.append(
            portfolio_return
        )

        # ============================================================
        # RİSK
        # ============================================================

        if len(self.return_history) >= 2:
            risk = float(
                np.std(self.return_history)
            )
        else:
            risk = 0.0

        # ============================================================
        # RISK-AWARE REWARD
        # ============================================================

        return_weight = 10.0
        risk_weight = 0.5
        transaction_weight = 0.5

        reward = (
            return_weight * portfolio_return
            - risk_weight * risk
            + transaction_weight * transaction_cost
        )

        # ============================================================
        # REWARD SINIRI
        # ============================================================

        reward = float(
            np.clip(
                reward,
                -1.0,
                1.0
            )
        )

        # ============================================================
        # SONRAKİ GÜN
        # ============================================================

        self.current_step += 1

        terminated = False
        truncated = False
        termination_reason = None

        # ------------------------------------------------------------
        # 1 - Veri bitti
        # ------------------------------------------------------------

        if self.current_step >= self.n_steps:

            terminated = True
            termination_reason = "End_of_Dataset"

        # ------------------------------------------------------------
        # 2 - Cash sıfır
        # ------------------------------------------------------------

        elif self.cash <= 0:

            terminated = True
            termination_reason = "Cash_Zero"

        # ------------------------------------------------------------
        # 3 - Portföy değeri çok düştü
        # ------------------------------------------------------------

        elif (
            self.portfolio_value
            <= self.initial_cash
            * self.min_portfolio_ratio
        ):

            terminated = True
            termination_reason = "Portfolio_Below_Limit"

        # ------------------------------------------------------------
        # 4 - Hedef getirildi
        # ------------------------------------------------------------

        elif (
            self.portfolio_value
            >= self.initial_cash
            * self.target_portfolio_ratio
        ):

            terminated = True
            termination_reason = "Target_Return_Reached"

        # ============================================================
        # SONRAKİ STATE
        # ============================================================

        if terminated:

            state = np.zeros(
                self.observation_space.shape,
                dtype=np.float32
            )

        else:

            state = self._get_state()

        # ============================================================
        # STEP INFORMATION
        # ============================================================

        info = {

            # --------------------------------------------------------
            # State
            # --------------------------------------------------------

            "state": [
                float(x)
                for x in state
            ],

            # --------------------------------------------------------
            # Scenario
            # --------------------------------------------------------

            "scenario": int(
                self.current_scenario
            ),

            "scenario_name": self.scenario_names[
                int(self.current_scenario)
            ],

            # --------------------------------------------------------
            # Date
            # --------------------------------------------------------

            "date": str(
                today["Date"]
            ),

            # --------------------------------------------------------
            # Step
            # --------------------------------------------------------

            "step": self.current_step,

            # --------------------------------------------------------
            # Action
            # --------------------------------------------------------

            "action": int(action),

            "action_name": self.action_names[
                action
            ],

            # --------------------------------------------------------
            # Trade Details
            # --------------------------------------------------------

            "trade_amount": float(
                trade_amount
            ),

            "trade_source": trade_source,

            "trade_target": trade_target,

            "full_liquidation": bool(
                full_liquidation
            ),

            # --------------------------------------------------------
            # Reward
            # --------------------------------------------------------

            "reward": float(
                reward
            ),

            "portfolio_return": float(
                portfolio_return
            ),

            "risk": float(
                risk
            ),

            # --------------------------------------------------------
            # Market
            # --------------------------------------------------------

            "daily_return": float(
                daily_return
            ),

            "fund_return": float(
                fund_return
            ),

            "deviation": float(
                deviation
            ),

            "portfolio_multiplier": float(
                portfolio_multiplier
            ),

            # --------------------------------------------------------
            # Portfolio
            # --------------------------------------------------------

            "portfolio_value": float(
                self.portfolio_value
            ),

            "previous_portfolio": float(
                previous_value
            ),

            # --------------------------------------------------------
            # Portfolio Allocation
            # --------------------------------------------------------

            "portfolio_allocation": {

                "stock": float(
                    self.stock_weight
                ),

                "repo": float(
                    self.repo_weight
                ),

                "collateral": float(
                    self.collateral_weight
                ),

                "fund": float(
                    self.fund_weight
                ),

                # Public versiyonda bu değerler
                # tamamen mock/sentetik değerlerdir.

                "stock_breakdown": {
                    ticker: float(weight)
                    for ticker, weight
                    in self.stock_breakdown.items()
                }
            },

            # --------------------------------------------------------
            # Portfolio Weights
            # --------------------------------------------------------

            "stock_weight": float(
                self.stock_weight
            ),

            "repo_weight": float(
                self.repo_weight
            ),

            "collateral_weight": float(
                self.collateral_weight
            ),

            "fund_weight": float(
                self.fund_weight
            ),

            # --------------------------------------------------------
            # Cost
            # --------------------------------------------------------

            "transaction_cost": float(
                transaction_cost
            ),

            # --------------------------------------------------------
            # Episode Status
            # --------------------------------------------------------

            "terminated": terminated,

            "truncated": truncated,

            "termination_reason": termination_reason
        }

        return (
            state,
            reward,
            terminated,
            truncated,
            info
        )

    # ================================================================
    # RENDER
    # ================================================================

    def render(self):

        print("=" * 50)

        print(
            f"Scenario        : "
            f"{self.current_scenario}"
        )

        print(
            f"Step            : "
            f"{self.current_step}/{self.n_steps}"
        )

        print(
            f"Portföy Değeri  : "
            f"{self.portfolio_value:.2f} TL"
        )

        print(
            f"Nakit           : "
            f"{self.cash:.2f} TL"
        )

        print("\nPortföy Ağırlıkları")

        print(
            f"HS (Hisse)      : "
            f"%{self.stock_weight * 100:.2f}"
        )

        print(
            f"TPP (Repo)      : "
            f"%{self.repo_weight * 100:.2f}"
        )

        print(
            f"T (Teminat)     : "
            f"%{self.collateral_weight * 100:.2f}"
        )

        print(
            f"YYF (Fon)       : "
            f"%{self.fund_weight * 100:.2f}"
        )

        print("\nState:")

        print(
            self._get_state()
        )
