import { Link } from 'react-router-dom'
import { AuthLayout } from './AuthLayout'
import { AuthForm } from './AuthForm'

export default function SignUp() {
  return (
    <AuthLayout
      title="Start monitoring the planet"
      subtitle={
        <>
          Already have an account?{' '}
          <Link to="/signin" className="font-semibold text-cv-primary hover:text-cv-primary-hover">
            Sign in
          </Link>
        </>
      }
    >
      <AuthForm mode="signup" />
    </AuthLayout>
  )
}
