class Card:
    def __init__(self, rank):
        self.rank = str(rank).upper()

    def get_value(self):
        if self.rank in ['J', 'Q', 'K', '10', 'JACK', 'QUEEN', 'KING', 'TEN']:
            return 10
        if self.rank in ['A', 'ACE']:
            return 11
        try:
            return int(self.rank)
        except ValueError:
            return 0

class BlackjackLogic:
    @staticmethod
    def calculate_score_and_is_soft(ranks):
        if not ranks:
            return 0, False

        card_values = [Card(r).get_value() for r in ranks if Card(r).get_value() > 0]
        score = sum(card_values)

        aces = sum(1 for r in ranks if str(r).upper() in ['A', 'ACE'])

        while score > 21 and aces > 0:
            score -= 10
            aces -= 1

        is_soft = (aces > 0 and score <= 21)
        return score, is_soft

    @staticmethod
    def get_hint(p_ranks, d_ranks):
        p_score, p_soft = BlackjackLogic.calculate_score_and_is_soft(p_ranks)

        if p_score == 0: return "WAIT"
        if p_score > 21: return "BUST"

        if not d_ranks: return "WAIT"

        first_d_rank = str(d_ranks[0]).upper()
        first_d_rank = 'A' if first_d_rank == 'ACE' else first_d_rank
        d_up_card_val = Card(first_d_rank).get_value()

        p_ranks_clean = ['A' if str(r).upper() == 'ACE' else str(r).upper() for r in p_ranks]

        if len(p_ranks_clean) == 2 and p_ranks_clean[0] == p_ranks_clean[1]:
            pair_card = p_ranks_clean[0]
            if pair_card in ['A', '8']:
                return "SPLIT"
            if pair_card in ['9'] and d_up_card_val not in [7, 10, 11]:
                return "SPLIT"
            if pair_card in ['7', '3', '2'] and d_up_card_val <= 7:
                return "SPLIT"
            if pair_card in ['6'] and d_up_card_val <= 6:
                return "SPLIT"

        if p_soft:
            if p_score >= 19: return "STAND"
            if p_score == 18:
                if d_up_card_val in [2, 7, 8]: return "STAND"
                if d_up_card_val in [3, 4, 5, 6] and len(p_ranks_clean) == 2: return "DOUBLE"
                return "HIT"
            if len(p_ranks_clean) == 2 and d_up_card_val in [5, 6]: return "DOUBLE"
            if p_score == 17 and len(p_ranks_clean) == 2 and d_up_card_val == 4: return "DOUBLE"
            return "HIT"

        if p_score >= 17: return "STAND"
        if 12 <= p_score <= 16:
            if 2 <= d_up_card_val <= 6:
                if p_score == 12 and d_up_card_val in [2, 3]: return "HIT"
                return "STAND"
            return "HIT"

        if len(p_ranks_clean) == 2:
            if p_score == 11: return "DOUBLE"
            if p_score == 10 and d_up_card_val <= 9: return "DOUBLE"
            if p_score == 9 and d_up_card_val in [3, 4, 5, 6]: return "DOUBLE"

        return "HIT"