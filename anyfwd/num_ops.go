package anyfwd

import "github.com/unixpickle/anyvec"

// NumOps implements anyvec.NumOps for Numeric values.
type NumOps struct {
	ValueOps anyvec.NumOps
}

// Add adds two Numerics.
func (n NumOps) Add(n1, n2 anyvec.Numeric) anyvec.Numeric {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	sum := Numeric{
		Value: n.ValueOps.Add(num1.Value, num2.Value),
	}
	for i, g1 := range num1.Grad {
		g2 := num2.Grad[i]
		sum.Grad = append(sum.Grad, n.ValueOps.Add(g1, g2))
	}
	return sum
}

// Sub subtracts two Numerics.
func (n NumOps) Sub(n1, n2 anyvec.Numeric) anyvec.Numeric {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	sum := Numeric{
		Value: n.ValueOps.Sub(num1.Value, num2.Value),
	}
	for i, g1 := range num1.Grad {
		g2 := num2.Grad[i]
		sum.Grad = append(sum.Grad, n.ValueOps.Sub(g1, g2))
	}
	return sum
}

// Mul multiplies two Numerics.
func (n NumOps) Mul(n1, n2 anyvec.Numeric) anyvec.Numeric {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	sum := Numeric{
		Value: n.ValueOps.Mul(num1.Value, num2.Value),
	}
	for i, g1 := range num1.Grad {
		g2 := num2.Grad[i]
		productRule := n.ValueOps.Add(
			n.ValueOps.Mul(g1, num2.Value),
			n.ValueOps.Mul(num1.Value, g2),
		)
		sum.Grad = append(sum.Grad, productRule)
	}
	return sum
}

// Div divides two Numerics.
func (n NumOps) Div(n1, n2 anyvec.Numeric) anyvec.Numeric {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	sum := Numeric{
		Value: n.ValueOps.Div(num1.Value, num2.Value),
	}
	for i, g1 := range num1.Grad {
		g2 := num2.Grad[i]
		quotientRule := n.ValueOps.Sub(
			n.ValueOps.Div(g1, num2.Value),
			n.ValueOps.Div(
				n.ValueOps.Mul(num1.Value, g2),
				n.ValueOps.Mul(num2.Value, num2.Value),
			),
		)
		sum.Grad = append(sum.Grad, quotientRule)
	}
	return sum
}

// Identical checks if two Numerics are exactly identical,
// including their derivatives.
func (n NumOps) Identical(n1, n2 anyvec.Numeric) bool {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	if !n.ValueOps.Identical(num1.Value, num2.Value) {
		return false
	}
	for i, g1 := range num1.Grad {
		g2 := num2.Grad[i]
		if !n.ValueOps.Identical(g1, g2) {
			return false
		}
	}
	return true
}

// Equal checks if two Numerics have the same value.
func (n NumOps) Equal(n1, n2 anyvec.Numeric) bool {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	return n.ValueOps.Equal(num1.Value, num2.Value)
}

// Less checks if one Numeric is less than another.
func (n NumOps) Less(n1, n2 anyvec.Numeric) bool {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	return n.ValueOps.Less(num1.Value, num2.Value)
}

// Greater checks if one Numeric is greater than another.
func (n NumOps) Greater(n1, n2 anyvec.Numeric) bool {
	num1 := n1.(Numeric)
	num2 := n2.(Numeric)
	return n.ValueOps.Greater(num1.Value, num2.Value)
}
