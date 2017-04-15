package anyfwd

import "github.com/unixpickle/anyvec"

// Complement computes 1 - v.
func (v *Vector) Complement() {
	anyvec.Complement(v.Values)
	for _, grad := range v.Jacobian {
		grad.Scale(grad.Creator().MakeNumeric(-1))
	}
}

// GreaterThan performs a component-wise comparison.
//
// The result is considered constant and its derivatives
// are all 0.
func (v *Vector) GreaterThan(n anyvec.Numeric) {
	v.comparison(n, anyvec.GreaterThan)
}

// LessThan performs a component-wise comparison.
//
// The result is considered constant and its derivatives
// are all 0.
func (v *Vector) LessThan(n anyvec.Numeric) {
	v.comparison(n, anyvec.LessThan)
}

// EqualTo performs a component-wise comparison.
//
// The result is considered constant and its derivatives
// are all 0.
func (v *Vector) EqualTo(n anyvec.Numeric) {
	v.comparison(n, anyvec.EqualTo)
}

func (v *Vector) comparison(n anyvec.Numeric, f func(v anyvec.Vector, n anyvec.Numeric)) {
	f(v.Values, v.convertNum(n).Value)
	v.clearJacobian()
}
