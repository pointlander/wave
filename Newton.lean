-- Copyright 2026 The Wave Authors. All rights reserved.
-- Use of this source code is governed by a BSD-style
-- license that can be found in the LICENSE file.

/-
This file records the retired charged score. The simulator now scores
the von Neumann entropy of the discrete Fourier transform of Ψ and does not
add a departure charge.

The Newtonian proposal is the reality that charged score keeps when no
rival's drop in block entropy exceeds the departure charge.

`departure` is the mean squared residual against the proposal. `score`
adds `λ` times that residual to the entropy. A block QFT, modeled here
as the same orthogonal map on the offset inside every block, preserves
the trace and the determinant of the Gram matrix, so every entropy that
depends only on those two invariants is unchanged.
-/

theorem sq_nonneg (x : Rat) : 0 ≤ x * x := by
  rcases Rat.nonneg_total x with hx | hx
  · exact Rat.mul_nonneg hx hx
  · have h := Rat.mul_nonneg hx hx
    rwa [Rat.neg_mul, Rat.mul_neg, Rat.neg_neg] at h

theorem sq_pos {x : Rat} (h : x ≠ 0) : 0 < x * x := by
  refine Rat.lt_of_le_of_ne (sq_nonneg x) ?_
  intro hz
  exact h ((Rat.mul_eq_zero.mp hz.symm).elim id id)

theorem lt_of_lt_of_le {a b c : Rat} (hab : a < b) (hbc : b ≤ c) : a < c := by
  refine Rat.lt_of_le_of_ne (Rat.le_trans (Rat.le_of_lt hab) hbc) ?_
  intro hac
  exact (Rat.not_le.mpr hab) (hac ▸ hbc)

theorem add_pos_of_pos_of_nonneg {a b : Rat} (ha : 0 < a) (hb : 0 ≤ b) : 0 < a + b := by
  have hle : a ≤ a + b := by
    have := (Rat.add_le_add_left (c := a)).2 hb
    simpa [Rat.add_zero] using this
  exact lt_of_lt_of_le ha hle

theorem add_pos_of_nonneg_of_pos {a b : Rat} (ha : 0 ≤ a) (hb : 0 < b) : 0 < a + b := by
  have hle : b ≤ a + b := by
    have := (Rat.add_le_add_right (c := b)).2 ha
    simpa [Rat.zero_add] using this
  exact lt_of_lt_of_le hb hle

theorem sub_self (a : Rat) : a - a = 0 := by
  rw [Rat.sub_eq_add_neg, Rat.add_comm, Rat.neg_add_cancel]

theorem sub_eq_zero {a b : Rat} (h : a - b = 0) : a = b := by
  have h' := congrArg (fun z => z + b) h
  rw [Rat.sub_eq_add_neg, Rat.add_assoc, Rat.neg_add_cancel, Rat.add_zero, Rat.zero_add] at h'
  exact h'

theorem zero_div (a : Rat) : 0 / a = 0 := by
  rw [Rat.div_def, Rat.zero_mul]

theorem div_pos {a b : Rat} (ha : 0 < a) (hb : 0 < b) : 0 < a / b := by
  rw [Rat.div_def]
  exact Rat.mul_pos ha (Rat.inv_pos.mpr hb)

def sumSq : List Rat → Rat
  | [] => 0
  | x :: xs => x * x + sumSq xs

theorem sumSq_nonneg : (xs : List Rat) → 0 ≤ sumSq xs
  | [] => Rat.le_refl
  | x :: xs => Rat.add_nonneg (sq_nonneg x) (sumSq_nonneg xs)

theorem sumSq_pos : (xs : List Rat) → (∃ x, x ∈ xs ∧ x ≠ 0) → 0 < sumSq xs
  | [], h => by
    rcases h with ⟨_, hx, _⟩
    cases hx
  | y :: ys, h => by
    rw [sumSq]
    rcases h with ⟨x, hx, hne⟩
    cases List.mem_cons.mp hx with
    | inl heq =>
      subst heq
      exact add_pos_of_pos_of_nonneg (sq_pos hne) (sumSq_nonneg ys)
    | inr hys =>
      exact add_pos_of_nonneg_of_pos (sq_nonneg y) (sumSq_pos ys ⟨x, hys, hne⟩)

theorem sumSq_zeros : (n : Nat) → sumSq (List.replicate n (0 : Rat)) = 0
  | 0 => rfl
  | n + 1 => by
    rw [List.replicate_succ, sumSq, sumSq_zeros n, Rat.mul_zero, Rat.zero_add]

/-- Coordinate residuals of `ψ` against `ref`. The Newtonian proposal is `ref`. -/
def residuals : List Rat → List Rat → List Rat
  | x :: xs, y :: ys => (x - y) :: residuals xs ys
  | _, _ => []

theorem residuals_self : (xs : List Rat) → residuals xs xs = List.replicate xs.length 0
  | [] => rfl
  | x :: xs => by
    rw [residuals, residuals_self xs, sub_self, List.length_cons, List.replicate_succ]

theorem residuals_length {ψ ref : List Rat} (h : ψ.length = ref.length) :
    (residuals ψ ref).length = ψ.length := by
  induction ψ generalizing ref with
  | nil =>
    cases ref with
    | nil => rfl
    | cons _ _ => simp at h
  | cons x xs ih =>
    cases ref with
    | nil => simp at h
    | cons y ys =>
      rw [residuals, List.length_cons, List.length_cons]
      exact congrArg Nat.succ (ih (by simpa using h))

theorem residuals_ne_of_ne {ψ ref : List Rat} (hlen : ψ.length = ref.length) (hdiff : ψ ≠ ref) :
    ∃ r, r ∈ residuals ψ ref ∧ r ≠ 0 := by
  induction ψ generalizing ref with
  | nil =>
    cases ref with
    | nil => exact absurd rfl hdiff
    | cons _ _ => simp at hlen
  | cons x xs ih =>
    cases ref with
    | nil => simp at hlen
    | cons y ys =>
      have hlen' : xs.length = ys.length := by simpa using hlen
      by_cases hxy : x = y
      · subst hxy
        have htail : xs ≠ ys := by
          intro hxs
          exact hdiff (congrArg (List.cons x) hxs)
        rcases ih hlen' htail with ⟨r, hr, hr0⟩
        exact ⟨r, List.mem_cons_of_mem _ hr, hr0⟩
      · refine ⟨x - y, List.mem_cons_self, ?_⟩
        intro hsub
        exact hxy (sub_eq_zero hsub)

/-- Mean squared distance from `ref`, in the same units as the coordinates. -/
def departure (ψ ref : List Rat) : Rat :=
  let d := residuals ψ ref
  if d.isEmpty then 0 else sumSq d / (d.length : Rat)

theorem departure_self (xs : List Rat) : departure xs xs = 0 := by
  unfold departure
  rw [residuals_self]
  cases xs with
  | nil => simp
  | cons x xs =>
    rw [ite_eq_right]
    · rw [sumSq_zeros, zero_div]
    · simp

def score (H : List Rat → Rat) (lam : Rat) (ψ ref : List Rat) : Rat :=
  H ψ + lam * departure ψ ref

theorem sub_le_iff_le_add {a b c : Rat} : a - b ≤ c ↔ a ≤ b + c := by
  constructor
  · intro h
    rw [Rat.sub_eq_add_neg] at h
    have h' := (Rat.add_le_add_right (c := b)).2 h
    rw [Rat.add_assoc, Rat.neg_add_cancel, Rat.add_zero] at h'
    rwa [Rat.add_comm]
  · intro h
    rw [Rat.sub_eq_add_neg]
    have h' := (Rat.add_le_add_right (c := -b)).2 h
    rw [Rat.add_comm b c, Rat.add_assoc, Rat.add_comm b (-b), Rat.neg_add_cancel, Rat.add_zero] at h'
    exact h'

theorem sub_lt_iff_lt_add {a b c : Rat} : a - b < c ↔ a < b + c := by
  constructor
  · intro h
    rw [Rat.sub_eq_add_neg] at h
    have h' := (Rat.add_lt_add_right (c := b)).2 h
    rw [Rat.add_assoc, Rat.neg_add_cancel, Rat.add_zero] at h'
    rwa [Rat.add_comm]
  · intro h
    rw [Rat.sub_eq_add_neg]
    have h' := (Rat.add_lt_add_right (c := -b)).2 h
    rw [Rat.add_comm b c, Rat.add_assoc, Rat.add_comm b (-b), Rat.neg_add_cancel, Rat.add_zero] at h'
    exact h'

theorem charge_iff (H : List Rat → Rat) (lam : Rat) (ref ψ : List Rat) :
    H ref - H ψ ≤ lam * departure ψ ref ↔
      score H lam ref ref ≤ score H lam ψ ref := by
  unfold score
  rw [departure_self, Rat.mul_zero, Rat.add_zero]
  exact sub_le_iff_le_add

theorem charge_iff_strict (H : List Rat → Rat) (lam : Rat) (ref ψ : List Rat) :
    H ref - H ψ < lam * departure ψ ref ↔
      score H lam ref ref < score H lam ψ ref := by
  unfold score
  rw [departure_self, Rat.mul_zero, Rat.add_zero]
  exact sub_lt_iff_lt_add

/-- The proposal is scored first. A later frame replaces it only on a strict improvement. -/
def pick (H : List Rat → Rat) (lam : Rat) (ref best : List Rat) (bestS : Rat) :
    List (List Rat) → List Rat
  | [] => best
  | ψ :: rest =>
    let s := score H lam ψ ref
    if s < bestS then pick H lam ref ψ s rest else pick H lam ref best bestS rest

theorem pick_eq_of_no_improvement (H : List Rat → Rat) (lam : Rat) (ref : List Rat)
    (rest : List (List Rat))
    (h : ∀ ψ ∈ rest, ¬ score H lam ψ ref < score H lam ref ref) :
    pick H lam ref ref (score H lam ref ref) rest = ref := by
  induction rest with
  | nil => rfl
  | cons ψ rest ih =>
    unfold pick
    rw [ite_eq_right (h ψ List.mem_cons_self)]
    exact ih fun φ hφ => h φ (List.mem_cons_of_mem _ hφ)

/-- If every candidate's entropy drop is within the departure charge, the
search returns the Newtonian proposal. -/
theorem newton_is_selected (H : List Rat → Rat) (lam : Rat) (ref : List Rat)
    (rest : List (List Rat))
    (hcharge : ∀ ψ ∈ ref :: rest, H ref - H ψ ≤ lam * departure ψ ref) :
    pick H lam ref ref (score H lam ref ref) rest = ref := by
  refine pick_eq_of_no_improvement H lam ref rest ?_
  intro ψ hψ
  rw [Rat.not_lt]
  exact (charge_iff H lam ref ψ).1 (hcharge ψ (List.mem_cons_of_mem _ hψ))

theorem departure_pos {ψ ref : List Rat} (hlen : ψ.length = ref.length) (hdiff : ψ ≠ ref) :
    0 < departure ψ ref := by
  rcases residuals_ne_of_ne hlen hdiff with ⟨r, hr, hr0⟩
  have hsum := sumSq_pos (residuals ψ ref) ⟨r, hr, hr0⟩
  have hlenr : 0 < (residuals ψ ref).length := List.length_pos_of_mem hr
  have hempty : (residuals ψ ref).isEmpty = false := by
    cases h : (residuals ψ ref).isEmpty with
    | false => rfl
    | true =>
      have hnil : residuals ψ ref = [] := by simpa [List.isEmpty_iff] using h
      exact absurd (by rw [hnil]; rfl) (Nat.ne_of_gt hlenr)
  simp [departure, hempty]
  exact div_pos hsum (Rat.natCast_pos.mpr hlenr)

/-
A block QFT applies one unitary to the offset inside every block and does
not mix blocks. On a real two-dimensional offset that unitary is an
orthogonal matrix. The Gram determinant is the square of the cross product
of the two block vectors, and an orthogonal map multiplies the cross
product by `det U ∈ {±1}`.
-/

structure Amp where
  b0l0 : Rat
  b0l1 : Rat
  b1l0 : Rat
  b1l1 : Rat

structure Orth where
  a : Rat
  b : Rat
  c : Rat
  d : Rat
  hcol0 : a * a + c * c = 1
  hcol1 : b * b + d * d = 1
  horth : a * b + c * d = 0

def applyLocal (U : Orth) (ψ : Amp) : Amp where
  b0l0 := U.a * ψ.b0l0 + U.b * ψ.b0l1
  b0l1 := U.c * ψ.b0l0 + U.d * ψ.b0l1
  b1l0 := U.a * ψ.b1l0 + U.b * ψ.b1l1
  b1l1 := U.c * ψ.b1l0 + U.d * ψ.b1l1

def g00 (ψ : Amp) : Rat := ψ.b0l0 * ψ.b0l0 + ψ.b1l0 * ψ.b1l0
def g01 (ψ : Amp) : Rat := ψ.b0l0 * ψ.b0l1 + ψ.b1l0 * ψ.b1l1
def g11 (ψ : Amp) : Rat := ψ.b0l1 * ψ.b0l1 + ψ.b1l1 * ψ.b1l1
def gramTrace (ψ : Amp) : Rat := g00 ψ + g11 ψ
def gramDet (ψ : Amp) : Rat := g00 ψ * g11 ψ - g01 ψ * g01 ψ
def cross (ψ : Amp) : Rat := ψ.b0l0 * ψ.b1l1 - ψ.b0l1 * ψ.b1l0

def entropy (E : Rat → Rat → Rat) (ψ : Amp) : Rat := E (gramTrace ψ) (gramDet ψ)

theorem detSq_orth {a b c d : Rat}
    (h1 : a * a + c * c = 1) (h2 : b * b + d * d = 1) (h3 : a * b + c * d = 0) :
    (a * d - b * c) * (a * d - b * c) = 1 := by
  grind

theorem gramDet_eq_cross_sq (ψ : Amp) : gramDet ψ = cross ψ * cross ψ := by
  grind [gramDet, g00, g01, g11, cross]

theorem cross_applyLocal (U : Orth) (ψ : Amp) :
    cross (applyLocal U ψ) = (U.a * U.d - U.b * U.c) * cross ψ := by
  grind [cross, applyLocal]

theorem gramTrace_apply (U : Orth) (ψ : Amp) :
    gramTrace (applyLocal U ψ) = gramTrace ψ := by
  grind [gramTrace, g00, g11, applyLocal, U.hcol0, U.hcol1, U.horth]

theorem gramDet_apply (U : Orth) (ψ : Amp) : gramDet (applyLocal U ψ) = gramDet ψ := by
  rw [gramDet_eq_cross_sq, gramDet_eq_cross_sq, cross_applyLocal]
  have hdet := detSq_orth U.hcol0 U.hcol1 U.horth
  grind

/-- A block-local orthogonal transform leaves every spectral entropy unchanged. -/
theorem qft_preserves_entropy (E : Rat → Rat → Rat) (U : Orth) (ψ : Amp) :
    entropy E (applyLocal U ψ) = entropy E ψ := by
  unfold entropy
  rw [gramTrace_apply, gramDet_apply]
