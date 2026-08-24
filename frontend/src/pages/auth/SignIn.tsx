import { Link } from 'react-router-dom'
import { AuthLayout } from './AuthLayout'
import { AuthForm } from './AuthForm'

export default function SignIn() {
  return (
    <AuthLayout
      title="Welcome back"
      subtitle={
        <>
          New to ClimateVision?{' '}
          <Link to="/signup" className="font-semibold text-cv-primary hover:text-cv-primary-hover">
            Create a free account
          </Link>
        </>
      }
    >
      <AuthForm mode="signin" />
    </AuthLayout>
  )
}
