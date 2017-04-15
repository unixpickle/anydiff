package anyfwd

import "github.com/unixpickle/anyvec"

// Mapper is an anyvec.Mapper which can be applied to
// *Vector instances.
type Mapper struct {
	CreatorPtr  *Creator
	ValueMapper anyvec.Mapper
}

// Creator returns m.CreatorPtr.
func (m *Mapper) Creator() anyvec.Creator {
	return m.CreatorPtr
}

// InSize returns the value mapper's input size.
func (m *Mapper) InSize() int {
	return m.ValueMapper.InSize()
}

// OutSize returns the value mapper's output size.
func (m *Mapper) OutSize() int {
	return m.ValueMapper.OutSize()
}

// Map applies the map operation.
func (m *Mapper) Map(in, out anyvec.Vector) {
	vin := in.(*Vector)
	vout := out.(*Vector)
	if len(vin.Jacobian) != len(vout.Jacobian) {
		panic(badJacobianErr)
	}
	m.ValueMapper.Map(vin.Values, vout.Values)
	for i, x := range vin.Jacobian {
		m.ValueMapper.Map(x, vout.Jacobian[i])
	}
}

// MapTranspose applies the transposed map operation.
func (m *Mapper) MapTranspose(in, out anyvec.Vector) {
	vin := in.(*Vector)
	vout := out.(*Vector)
	if len(vin.Jacobian) != len(vout.Jacobian) {
		panic(badJacobianErr)
	}
	m.ValueMapper.MapTranspose(vin.Values, vout.Values)
	for i, x := range vin.Jacobian {
		m.ValueMapper.MapTranspose(x, vout.Jacobian[i])
	}
}

// MapMax creates a *Mapper which selects the maximum
// element in each row of v.
func (v *Vector) MapMax(cols int) anyvec.Mapper {
	return &Mapper{
		CreatorPtr:  v.CreatorPtr,
		ValueMapper: anyvec.MapMax(v.Values, cols),
	}
}
